//===- ErrorHandler.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/ErrorHandler.h"

#include "lld/Common/Threads.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#endif

using namespace llvm;
using namespace lld;

// The functions defined in this file can be called from multiple threads,
// but outs() or errs() are not thread-safe. We protect them using a mutex.
static std::mutex Mu;

// Prints "\n" or does nothing, depending on Msg contents of
// the previous call of this function.
static void newline(raw_ostream *ErrorOS, const Twine &Msg) {
  // True if the previous error message contained "\n".
  // We want to separate multi-line error messages with a newline.
  static bool Flag;

  if (Flag)
    *ErrorOS << "\n";
  Flag = StringRef(Msg.str()).contains('\n');
}

ErrorHandler &lld::errorHandler() {
  static ErrorHandler Handler;
  return Handler;
}

void lld::exitLld(int Val) {
  // Delete any temporary file, while keeping the memory mapping open.
  if (errorHandler().OutputBuffer)
    errorHandler().OutputBuffer->discard();

  // Dealloc/destroy ManagedStatic variables before calling
  // _exit(). In a non-LTO build, this is a nop. In an LTO
  // build allows us to get the output of -time-passes.
  llvm_shutdown();

  outs().flush();
  errs().flush();
  _exit(Val);
}

void lld::diagnosticHandler(const DiagnosticInfo &DI) {
  SmallString<128> S;
  raw_svector_ostream OS(S);
  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);
  switch (DI.getSeverity()) {
  case DS_Error:
    error(S);
    break;
  case DS_Warning:
    warn(S);
    break;
  case DS_Remark:
  case DS_Note:
    message(S);
    break;
  }
}

void lld::checkError(Error E) {
  handleAllErrors(std::move(E),
                  [&](ErrorInfoBase &EIB) { error(EIB.message()); });
}

void ErrorHandler::print(StringRef S, raw_ostream::Colors C) {
  *ErrorOS << LogName << ": ";
  if (ColorDiagnostics) {
    ErrorOS->changeColor(C, true);
    *ErrorOS << S;
    ErrorOS->resetColor();
  } else {
    *ErrorOS << S;
  }
}

void ErrorHandler::log(const Twine &Msg) {
  if (Verbose) {
    std::lock_guard<std::mutex> Lock(Mu);
    *ErrorOS << LogName << ": " << Msg << "\n";
  }
}

void ErrorHandler::message(const Twine &Msg) {
  std::lock_guard<std::mutex> Lock(Mu);
  outs() << Msg << "\n";
  outs().flush();
}

void ErrorHandler::warn(const Twine &Msg) {
  if (FatalWarnings) {
    error(Msg);
    return;
  }

  std::lock_guard<std::mutex> Lock(Mu);
  newline(ErrorOS, Msg);
  print("warning: ", raw_ostream::MAGENTA);
  *ErrorOS << Msg << "\n";
}

void ErrorHandler::error(const Twine &Msg) {
  std::lock_guard<std::mutex> Lock(Mu);
  newline(ErrorOS, Msg);

  if (ErrorLimit == 0 || ErrorCount < ErrorLimit) {
    print("error: ", raw_ostream::RED);
    *ErrorOS << Msg << "\n";
  } else if (ErrorCount == ErrorLimit) {
    print("error: ", raw_ostream::RED);
    *ErrorOS << ErrorLimitExceededMsg << "\n";
    if (ExitEarly)
      exitLld(1);
  }

  ++ErrorCount;
}

void ErrorHandler::fatal(const Twine &Msg) {
  error(Msg);
  exitLld(1);
}
