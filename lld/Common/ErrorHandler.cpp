//===- ErrorHandler.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/ErrorHandler.h"

#include "lld/Common/CommonLinkerContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;
using namespace lld;

static StringRef getSeparator(const Twine &msg) {
  if (StringRef(msg.str()).contains('\n'))
    return "\n";
  return "";
}

namespace {

struct SourceLocation {
  StringRef file;
  StringRef line;
};

static std::optional<SourceLocation>
parseFileLine(StringRef text, bool requireClosingParen = false) {
  StringRef token = text.take_while([](char c) { return !isSpace(c); });
  for (size_t colon = token.rfind(':'); colon != StringRef::npos;) {
    size_t lineBegin = colon + 1;
    size_t lineEnd = lineBegin;
    while (lineEnd != token.size() && isDigit(token[lineEnd]))
      ++lineEnd;

    if (colon != 0 && lineEnd != lineBegin &&
        (!requireClosingParen ||
         (lineEnd != token.size() && token[lineEnd] == ')')))
      return SourceLocation{token.take_front(colon),
                            token.slice(lineBegin, lineEnd)};

    if (colon == 0)
      break;
    colon = token.rfind(':', colon - 1);
  }
  return std::nullopt;
}

static std::optional<SourceLocation>
parseParenthesizedFileLine(StringRef text) {
  for (size_t open = text.rfind('('); open != StringRef::npos;) {
    // The regexp required at least one character before the opening
    // parenthesis. Its '.' did not match carriage returns.
    if (open != 0 && !text.take_front(open).contains('\r'))
      if (std::optional<SourceLocation> location =
              parseFileLine(text.drop_front(open + 1),
                            /*requireClosingParen=*/true))
        return location;

    if (open == 0)
      break;
    open = text.rfind('(', open - 1);
  }
  return std::nullopt;
}

static std::string formatLocation(SourceLocation location) {
  return (Twine(location.file) + "(" + location.line + ")").str();
}

static bool isUndefinedSymbol(StringRef line) {
  if (line.contains('\r'))
    return false;
  if (!line.consume_front("undefined "))
    return false;
  if (line.starts_with("symbol:"))
    return true;

  auto [kind, rest] = line.split(' ');
  return !kind.empty() && none_of(kind, [](char c) { return isSpace(c); }) &&
         rest.starts_with("symbol:");
}

static std::optional<SourceLocation>
findReferencedLocation(ArrayRef<StringRef> lines, bool parenthesized) {
  for (size_t i = 0; i + 2 < lines.size(); ++i) {
    if (lines[i + 1].contains('\r') ||
        !lines[i + 1].starts_with(">>> defined in "))
      continue;
    StringRef referenced = lines[i + 2];
    if (!referenced.consume_front(">>> referenced by "))
      continue;
    if (std::optional<SourceLocation> location =
            parenthesized ? parseParenthesizedFileLine(referenced)
                          : parseFileLine(referenced))
      return location;
  }
  return std::nullopt;
}

static std::optional<SourceLocation> findUnclosedQuoteLocation(StringRef text) {
  constexpr StringLiteral suffix = ": unclosed quote";
  for (size_t suffixPos = text.find(suffix); suffixPos != StringRef::npos;
       suffixPos = text.find(suffix, suffixPos + 1)) {
    size_t lineEnd = suffixPos;
    size_t lineBegin = lineEnd;
    while (lineBegin != 0 && isDigit(text[lineBegin - 1]))
      --lineBegin;
    if (lineBegin == lineEnd || lineBegin == 0 || text[lineBegin - 1] != ':')
      continue;

    size_t fileEnd = lineBegin - 1;
    size_t fileBegin = fileEnd;
    while (fileBegin != 0 && !isSpace(text[fileBegin - 1]))
      --fileBegin;
    if (fileBegin != fileEnd)
      return SourceLocation{text.slice(fileBegin, fileEnd),
                            text.slice(lineBegin, lineEnd)};
  }
  return std::nullopt;
}

static bool isDefinedAtLine(StringRef line) {
  return line.consume_front(">>> defined at ") &&
         parseFileLine(line).has_value();
}

} // namespace

ErrorHandler::~ErrorHandler() {
  if (cleanupCallback)
    cleanupCallback();
}

void ErrorHandler::initialize(llvm::raw_ostream &stdoutOS,
                              llvm::raw_ostream &stderrOS, bool exitEarly,
                              bool disableOutput) {
  this->stdoutOS = &stdoutOS;
  this->stderrOS = &stderrOS;
  stderrOS.enable_colors(stderrOS.has_colors());
  this->exitEarly = exitEarly;
  this->disableOutput = disableOutput;
}

void ErrorHandler::flushStreams() {
  std::lock_guard<std::mutex> lock(mu);
  outs().flush();
  errs().flush();
}

ErrorHandler &lld::errorHandler() { return context().e; }

void lld::error(const Twine &msg) { errorHandler().error(msg); }
void lld::error(const Twine &msg, ErrorTag tag, ArrayRef<StringRef> args) {
  errorHandler().error(msg, tag, args);
}
void lld::fatal(const Twine &msg) { errorHandler().fatal(msg); }
void lld::log(const Twine &msg) { errorHandler().log(msg); }
void lld::message(const Twine &msg, llvm::raw_ostream &s) {
  errorHandler().message(msg, s);
}
void lld::warn(const Twine &msg) { errorHandler().warn(msg); }
uint64_t lld::errorCount() { return errorHandler().errorCount; }

raw_ostream &lld::outs() {
  ErrorHandler &e = errorHandler();
  return e.outs();
}

raw_ostream &ErrorHandler::outs() {
  if (disableOutput)
    return llvm::nulls();
  return stdoutOS ? *stdoutOS : llvm::outs();
}

raw_ostream &ErrorHandler::errs() {
  if (disableOutput)
    return llvm::nulls();
  return stderrOS ? *stderrOS : llvm::errs();
}

void lld::exitLld(int val) {
  if (hasContext()) {
    ErrorHandler &e = errorHandler();
    // Delete any temporary file, while keeping the memory mapping open.
    if (e.outputBuffer)
      e.outputBuffer->discard();
  }

  // Re-throw a possible signal or exception once/if it was caught by
  // safeLldMain().
  CrashRecoveryContext::throwIfCrash(val);

  // Dealloc/destroy ManagedStatic variables before calling _exit().
  // In an LTO build, allows us to get the output of -time-passes.
  // Ensures that the thread pool for the parallel algorithms is stopped to
  // avoid intermittent crashes on Windows when exiting.
  if (!CrashRecoveryContext::GetCurrent())
    llvm_shutdown();

  if (hasContext())
    lld::errorHandler().flushStreams();

  // When running inside safeLldMain(), restore the control flow back to the
  // CrashRecoveryContext. Otherwise simply use _exit(), meanning no cleanup,
  // since we want to avoid further crashes on shutdown.
  llvm::sys::Process::Exit(val, /*NoCleanup=*/true);
}

void lld::diagnosticHandler(const DiagnosticInfo &di) {
  SmallString<128> s;
  raw_svector_ostream os(s);
  DiagnosticPrinterRawOStream dp(os);

  // For an inline asm diagnostic, prepend the module name to get something like
  // "$module <inline asm>:1:5: ".
  if (auto *dism = dyn_cast<DiagnosticInfoSrcMgr>(&di))
    if (dism->isInlineAsmDiag())
      os << dism->getModuleName() << ' ';

  di.print(dp);
  switch (di.getSeverity()) {
  case DS_Error:
    error(s);
    break;
  case DS_Warning:
    warn(s);
    break;
  case DS_Remark:
  case DS_Note:
    message(s);
    break;
  }
}

void lld::checkError(Error e) {
  handleAllErrors(std::move(e),
                  [&](ErrorInfoBase &eib) { error(eib.message()); });
}

void lld::checkError(ErrorHandler &eh, Error e) {
  handleAllErrors(std::move(e),
                  [&](ErrorInfoBase &eib) { eh.error(eib.message()); });
}

// This is for --vs-diagnostics.
//
// Normally, lld's error message starts with argv[0]. Therefore, it usually
// looks like this:
//
//   ld.lld: error: ...
//
// This error message style is unfortunately unfriendly to Visual Studio
// IDE. VS interprets the first word of the first line as an error location
// and make it clickable, thus "ld.lld" in the above message would become a
// clickable text. When you click it, VS opens "ld.lld" executable file with
// a binary editor.
//
// As a workaround, we print out an error location instead of "ld.lld" if
// lld is running in VS diagnostics mode. As a result, error message will
// look like this:
//
//   src/foo.c(35): error: ...
//
// This function returns an error location string. An error location is
// extracted from one of lld's fixed diagnostic formats.
std::string ErrorHandler::getLocation(const Twine &msg) {
  if (!vsDiagnostics)
    return std::string(logName);

  std::string str = msg.str();
  SmallVector<StringRef, 8> lines;
  StringRef(str).split(lines, '\n');

  if (lines.size() >= 2 && isUndefinedSymbol(lines[0])) {
    StringRef referenced = lines[1];
    if (referenced.consume_front(">>> referenced by ")) {
      if (std::optional<SourceLocation> location =
              parseParenthesizedFileLine(referenced))
        return formatLocation(*location);
      if (std::optional<SourceLocation> location = parseFileLine(referenced))
        return formatLocation(*location);

      // This format is used when only an object-file location is available.
      if (lines[0].starts_with("undefined symbol:")) {
        StringRef beforeCarriageReturn = referenced.take_front(
            std::min(referenced.find('\r'), referenced.size()));
        if (size_t colon = beforeCarriageReturn.rfind(':');
            colon != StringRef::npos)
          return beforeCarriageReturn.take_front(colon).str();
      }
    }
  }

  if (lines.size() >= 3 && !lines[0].contains('\r') &&
      lines[0].starts_with("duplicate symbol: ")) {
    StringRef definedIn = lines[1];
    if (definedIn.consume_front(">>> defined in ") && !definedIn.empty() &&
        none_of(definedIn, [](char c) { return isSpace(c); }) &&
        lines[2].starts_with(">>> defined in"))
      return definedIn.str();

    StringRef definedAt = lines[1];
    if (definedAt.consume_front(">>> defined at ")) {
      if (std::optional<SourceLocation> location =
              parseParenthesizedFileLine(definedAt))
        return formatLocation(*location);
      if (std::optional<SourceLocation> location = parseFileLine(definedAt))
        return formatLocation(*location);
    }
  }

  if (std::optional<SourceLocation> location =
          findReferencedLocation(lines, /*parenthesized=*/true))
    return formatLocation(*location);
  if (std::optional<SourceLocation> location =
          findReferencedLocation(lines, /*parenthesized=*/false))
    return formatLocation(*location);
  if (std::optional<SourceLocation> location = findUnclosedQuoteLocation(str))
    return formatLocation(*location);

  return std::string(logName);
}

void ErrorHandler::reportDiagnostic(StringRef location, Colors c,
                                    StringRef diagKind, const Twine &msg) {
  SmallString<256> buf;
  raw_svector_ostream os(buf);
  os << sep << location << ": ";
  if (!diagKind.empty()) {
    if (errs().colors_enabled()) {
      os.enable_colors(true);
      os << c << diagKind << ": " << Colors::RESET;
    } else {
      os << diagKind << ": ";
    }
  }
  os << msg << '\n';
  errs() << buf;
  // If msg contains a newline, ensure that the next diagnostic is preceded by
  // a blank line separator.
  sep = getSeparator(msg);
}

void ErrorHandler::log(const Twine &msg) {
  if (!verbose || disableOutput)
    return;
  std::lock_guard<std::mutex> lock(mu);
  reportDiagnostic(logName, Colors::RESET, "", msg);
}

void ErrorHandler::message(const Twine &msg, llvm::raw_ostream &s) {
  if (disableOutput)
    return;
  std::lock_guard<std::mutex> lock(mu);
  s << msg << "\n";
  s.flush();
}

void ErrorHandler::warn(const Twine &msg) {
  if (fatalWarnings) {
    error(msg);
    return;
  }

  if (suppressWarnings)
    return;

  std::lock_guard<std::mutex> lock(mu);
  reportDiagnostic(getLocation(msg), Colors::MAGENTA, "warning", msg);
}

void ErrorHandler::error(const Twine &msg) {
  // If Visual Studio-style error message mode is enabled,
  // this particular error is printed out as two errors.
  if (vsDiagnostics) {
    std::string str = msg.str();
    SmallVector<StringRef, 5> lines;
    StringRef(str).split(lines, '\n');
    if (lines.size() == 5 && lines[0].starts_with("duplicate symbol: ") &&
        none_of(lines, [](StringRef line) { return line.contains('\r'); }) &&
        isDefinedAtLine(lines[1]) && lines[2].starts_with(">>>") &&
        isDefinedAtLine(lines[3]) && lines[4].starts_with(">>>")) {
      error(Twine(lines[0]) + "\n" + lines[1] + "\n" + lines[2]);
      error(Twine(lines[0]) + "\n" + lines[3] + "\n" + lines[4]);
      return;
    }
  }

  bool exit = false;
  {
    std::lock_guard<std::mutex> lock(mu);

    if (errorLimit == 0 || errorCount < errorLimit) {
      reportDiagnostic(getLocation(msg), Colors::RED, "error", msg);
    } else if (errorCount == errorLimit) {
      reportDiagnostic(logName, Colors::RED, "error", errorLimitExceededMsg);
      exit = exitEarly;
    }

    ++errorCount;
  }

  if (exit)
    exitLld(1);
}

void ErrorHandler::error(const Twine &msg, ErrorTag tag,
                         ArrayRef<StringRef> args) {
  if (errorHandlingScript.empty() || disableOutput) {
    error(msg);
    return;
  }
  SmallVector<StringRef, 4> scriptArgs;
  scriptArgs.push_back(errorHandlingScript);
  switch (tag) {
  case ErrorTag::LibNotFound:
    scriptArgs.push_back("missing-lib");
    break;
  case ErrorTag::SymbolNotFound:
    scriptArgs.push_back("undefined-symbol");
    break;
  }
  scriptArgs.insert(scriptArgs.end(), args.begin(), args.end());
  int res = llvm::sys::ExecuteAndWait(errorHandlingScript, scriptArgs);
  if (res == 0) {
    return error(msg);
  } else {
    // Temporarily disable error limit to make sure the two calls to error(...)
    // only count as one.
    uint64_t currentErrorLimit = errorLimit;
    errorLimit = 0;
    error(msg);
    errorLimit = currentErrorLimit;
    --errorCount;

    switch (res) {
    case -1:
      error("error handling script '" + errorHandlingScript +
            "' failed to execute");
      break;
    case -2:
      error("error handling script '" + errorHandlingScript +
            "' crashed or timeout");
      break;
    default:
      error("error handling script '" + errorHandlingScript +
            "' exited with code " + Twine(res));
    }
  }
}

void ErrorHandler::fatal(const Twine &msg) {
  error(msg);
  exitLld(1);
}

SyncStream::~SyncStream() {
  switch (level) {
  case DiagLevel::None:
    break;
  case DiagLevel::Log:
    e.log(buf);
    break;
  case DiagLevel::Msg:
    e.message(buf, e.outs());
    break;
  case DiagLevel::Warn:
    e.warn(buf);
    break;
  case DiagLevel::Err:
    e.error(buf);
    break;
  case DiagLevel::Fatal:
    e.fatal(buf);
    break;
  }
}
