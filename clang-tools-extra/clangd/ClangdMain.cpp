//===--- ClangdMain.cpp - clangd server loop ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdLSPServer.h"
#include "JSONRPCDispatcher.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include <iostream>
#include <memory>
#include <string>

using namespace clang;
using namespace clang::clangd;

static llvm::cl::opt<bool>
    RunSynchronously("run-synchronously",
                     llvm::cl::desc("parse on main thread"),
                     llvm::cl::init(false), llvm::cl::Hidden);

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "clangd");

  llvm::raw_ostream &Outs = llvm::outs();
  llvm::raw_ostream &Logs = llvm::errs();
  JSONOutput Out(Outs, Logs);

  // Change stdin to binary to not lose \r\n on windows.
  llvm::sys::ChangeStdinToBinary();

  ClangdLSPServer LSPServer(Out, RunSynchronously);
  LSPServer.run(std::cin);
}
