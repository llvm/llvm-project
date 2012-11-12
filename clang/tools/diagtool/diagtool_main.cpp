//===- diagtool_main.h - Entry point for invoking all diagnostic tools ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main function for diagtool.
//
//===----------------------------------------------------------------------===//

#include "DiagTool.h"

using namespace diagtool;

int main(int argc, char *argv[]) {
  if (argc > 1)
    if (DiagTool *tool = diagTools->getTool(argv[1]))
      return tool->run(argc - 2, &argv[2], llvm::outs());

  llvm::errs() << "usage: diagtool <command> [<args>]\n\n";
  diagTools->printCommands(llvm::errs());
  return 1;    
}
