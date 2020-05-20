//===--- QueryDriverDatabase.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some compiler drivers have implicit search mechanism for system headers.
// This compilation database implementation tries to extract that information by
// executing the driver in verbose mode. gcc-compatible drivers print something
// like:
// ....
// ....
// #include <...> search starts here:
//  /usr/lib/gcc/x86_64-linux-gnu/7/include
//  /usr/local/include
//  /usr/lib/gcc/x86_64-linux-gnu/7/include-fixed
//  /usr/include/x86_64-linux-gnu
//  /usr/include
// End of search list.
// ....
// ....
// This component parses that output and adds each path to command line args
// provided by Base, after prepending them with -isystem. Therefore current
// implementation would not work with a driver that is not gcc-compatible.
//
// First argument of the command line received from underlying compilation
// database is used as compiler driver path. Due to this arbitrary binary
// execution, this mechanism is not used by default and only executes binaries
// in the paths that are explicitly whitelisted by the user.

#include "GlobalCompilationDatabase.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Trace.h"
#include "clang/Driver/Types.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ScopedPrinter.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

std::vector<std::string> parseDriverOutput(llvm::StringRef Output) {
  std::vector<std::string> SystemIncludes;
  const char SIS[] = "#include <...> search starts here:";
  const char SIE[] = "End of search list.";
  llvm::SmallVector<llvm::StringRef, 8> Lines;
  Output.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  auto StartIt = llvm::find_if(
      Lines, [SIS](llvm::StringRef Line) { return Line.trim() == SIS; });
  if (StartIt == Lines.end()) {
    elog("System include extraction: start marker not found: {0}", Output);
    return {};
  }
  ++StartIt;
  const auto EndIt =
      llvm::find_if(llvm::make_range(StartIt, Lines.end()),
                    [SIE](llvm::StringRef Line) { return Line.trim() == SIE; });
  if (EndIt == Lines.end()) {
    elog("System include extraction: end marker missing: {0}", Output);
    return {};
  }

  for (llvm::StringRef Line : llvm::make_range(StartIt, EndIt)) {
    SystemIncludes.push_back(Line.trim().str());
    vlog("System include extraction: adding {0}", Line);
  }
  return SystemIncludes;
}

std::vector<std::string>
extractSystemIncludes(PathRef Driver, llvm::StringRef Lang,
                      llvm::ArrayRef<std::string> CommandLine,
                      llvm::Regex &QueryDriverRegex) {
  trace::Span Tracer("Extract system includes");
  SPAN_ATTACH(Tracer, "driver", Driver);
  SPAN_ATTACH(Tracer, "lang", Lang);

  if (!QueryDriverRegex.match(Driver)) {
    vlog("System include extraction: not whitelisted driver {0}", Driver);
    return {};
  }

  if (!llvm::sys::fs::exists(Driver)) {
    elog("System include extraction: {0} does not exist.", Driver);
    return {};
  }
  if (!llvm::sys::fs::can_execute(Driver)) {
    elog("System include extraction: {0} is not executable.", Driver);
    return {};
  }

  llvm::SmallString<128> StdErrPath;
  if (auto EC = llvm::sys::fs::createTemporaryFile("system-includes", "clangd",
                                                   StdErrPath)) {
    elog("System include extraction: failed to create temporary file with "
         "error {0}",
         EC.message());
    return {};
  }
  auto CleanUp = llvm::make_scope_exit(
      [&StdErrPath]() { llvm::sys::fs::remove(StdErrPath); });

  llvm::Optional<llvm::StringRef> Redirects[] = {
      {""}, {""}, llvm::StringRef(StdErrPath)};

  llvm::SmallVector<llvm::StringRef, 12> Args = {Driver, "-E", "-x",
                                                 Lang,   "-",  "-v"};

  // These flags will be preserved
  const llvm::StringRef FlagsToPreserve[] = {
      "-nostdinc", "--no-standard-includes", "-nostdinc++", "-nobuiltininc"};
  // Preserves these flags and their values, either as separate args or with an
  // equalsbetween them
  const llvm::StringRef ArgsToPreserve[] = {"--sysroot", "-isysroot"};

  for (size_t I = 0, E = CommandLine.size(); I < E; ++I) {
    llvm::StringRef Arg = CommandLine[I];
    if (llvm::any_of(FlagsToPreserve,
                     [&Arg](llvm::StringRef S) { return S == Arg; })) {
      Args.push_back(Arg);
    } else {
      const auto *Found =
          llvm::find_if(ArgsToPreserve, [&Arg](llvm::StringRef S) {
            return Arg.startswith(S);
          });
      if (Found == std::end(ArgsToPreserve))
        continue;
      Arg.consume_front(*Found);
      if (Arg.empty() && I + 1 < E) {
        Args.push_back(CommandLine[I]);
        Args.push_back(CommandLine[++I]);
      } else if (Arg.startswith("=")) {
        Args.push_back(CommandLine[I]);
      }
    }
  }

  if (int RC = llvm::sys::ExecuteAndWait(Driver, Args, /*Env=*/llvm::None,
                                         Redirects)) {
    elog("System include extraction: driver execution failed with return code: "
         "{0}. Args: ['{1}']",
         llvm::to_string(RC), llvm::join(Args, "', '"));
    return {};
  }

  auto BufOrError = llvm::MemoryBuffer::getFile(StdErrPath);
  if (!BufOrError) {
    elog("System include extraction: failed to read {0} with error {1}",
         StdErrPath, BufOrError.getError().message());
    return {};
  }

  auto Includes = parseDriverOutput(BufOrError->get()->getBuffer());
  log("System include extractor: successfully executed {0}, got includes: "
      "\"{1}\"",
      Driver, llvm::join(Includes, ", "));
  return Includes;
}

tooling::CompileCommand &
addSystemIncludes(tooling::CompileCommand &Cmd,
                  llvm::ArrayRef<std::string> SystemIncludes) {
  for (llvm::StringRef Include : SystemIncludes) {
    // FIXME(kadircet): This doesn't work when we have "--driver-mode=cl"
    Cmd.CommandLine.push_back("-isystem");
    Cmd.CommandLine.push_back(Include.str());
  }
  return Cmd;
}

/// Converts a glob containing only ** or * into a regex.
std::string convertGlobToRegex(llvm::StringRef Glob) {
  std::string RegText;
  llvm::raw_string_ostream RegStream(RegText);
  RegStream << '^';
  for (size_t I = 0, E = Glob.size(); I < E; ++I) {
    if (Glob[I] == '*') {
      if (I + 1 < E && Glob[I + 1] == '*') {
        // Double star, accept any sequence.
        RegStream << ".*";
        // Also skip the second star.
        ++I;
      } else {
        // Single star, accept any sequence without a slash.
        RegStream << "[^/]*";
      }
    } else {
      RegStream << llvm::Regex::escape(Glob.substr(I, 1));
    }
  }
  RegStream << '$';
  RegStream.flush();
  return RegText;
}

/// Converts a glob containing only ** or * into a regex.
llvm::Regex convertGlobsToRegex(llvm::ArrayRef<std::string> Globs) {
  assert(!Globs.empty() && "Globs cannot be empty!");
  std::vector<std::string> RegTexts;
  RegTexts.reserve(Globs.size());
  for (llvm::StringRef Glob : Globs)
    RegTexts.push_back(convertGlobToRegex(Glob));

  llvm::Regex Reg(llvm::join(RegTexts, "|"));
  assert(Reg.isValid(RegTexts.front()) &&
         "Created an invalid regex from globs");
  return Reg;
}

/// Extracts system includes from a trusted driver by parsing the output of
/// include search path and appends them to the commands coming from underlying
/// compilation database.
class QueryDriverDatabase : public GlobalCompilationDatabase {
public:
  QueryDriverDatabase(llvm::ArrayRef<std::string> QueryDriverGlobs,
                      std::unique_ptr<GlobalCompilationDatabase> Base)
      : QueryDriverRegex(convertGlobsToRegex(QueryDriverGlobs)),
        Base(std::move(Base)) {
    assert(this->Base);
    BaseChanged =
        this->Base->watch([this](const std::vector<std::string> &Changes) {
          OnCommandChanged.broadcast(Changes);
        });
  }

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override {
    auto Cmd = Base->getCompileCommand(File);
    if (!Cmd || Cmd->CommandLine.empty())
      return Cmd;

    llvm::StringRef Lang;
    for (size_t I = 0, E = Cmd->CommandLine.size(); I < E; ++I) {
      llvm::StringRef Arg = Cmd->CommandLine[I];
      if (Arg == "-x" && I + 1 < E)
        Lang = Cmd->CommandLine[I + 1];
      else if (Arg.startswith("-x"))
        Lang = Arg.drop_front(2).trim();
    }
    if (Lang.empty()) {
      llvm::StringRef Ext = llvm::sys::path::extension(File).trim('.');
      auto Type = driver::types::lookupTypeForExtension(Ext);
      if (Type == driver::types::TY_INVALID) {
        elog("System include extraction: invalid file type for {0}", Ext);
        return {};
      }
      Lang = driver::types::getTypeName(Type);
    }

    llvm::SmallString<128> Driver(Cmd->CommandLine.front());
    llvm::sys::fs::make_absolute(Cmd->Directory, Driver);
    auto Key = std::make_pair(Driver.str().str(), Lang.str());

    std::vector<std::string> SystemIncludes;
    {
      std::lock_guard<std::mutex> Lock(Mu);

      auto It = DriverToIncludesCache.find(Key);
      if (It != DriverToIncludesCache.end())
        SystemIncludes = It->second;
      else
        DriverToIncludesCache[Key] = SystemIncludes = extractSystemIncludes(
            Key.first, Key.second, Cmd->CommandLine, QueryDriverRegex);
    }

    return addSystemIncludes(*Cmd, SystemIncludes);
  }

  llvm::Optional<ProjectInfo> getProjectInfo(PathRef File) const override {
    return Base->getProjectInfo(File);
  }

private:
  mutable std::mutex Mu;
  // Caches includes extracted from a driver.
  mutable std::map<std::pair<std::string, std::string>,
                   std::vector<std::string>>
      DriverToIncludesCache;
  mutable llvm::Regex QueryDriverRegex;

  std::unique_ptr<GlobalCompilationDatabase> Base;
  CommandChanged::Subscription BaseChanged;
};
} // namespace

std::unique_ptr<GlobalCompilationDatabase>
getQueryDriverDatabase(llvm::ArrayRef<std::string> QueryDriverGlobs,
                       std::unique_ptr<GlobalCompilationDatabase> Base) {
  assert(Base && "Null base to SystemIncludeExtractor");
  if (QueryDriverGlobs.empty())
    return Base;
  return std::make_unique<QueryDriverDatabase>(QueryDriverGlobs,
                                               std::move(Base));
}

} // namespace clangd
} // namespace clang
