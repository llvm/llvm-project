//===-- PrintFlangFunctionNames.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small example Flang plugin to count/print Functions & Subroutines names.
// It walks the Parse Tree using a Visitor struct that has Post functions for
// FunctionStmt and SubroutineStmt to access the names of functions &
// subroutines. It also has Pre functions for FunctionSubprogram and
// SubroutineSubprogram so a Bool can be set to show that it is the definition
// of a function/subroutine, and not print those that are in an Interface.
// This plugin does not recognise Statement Functions or Module Procedures,
// which could be dealt with through StmtFunctionStmt and MpSubprogramStmt nodes
// respectively.
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"

using namespace Fortran::frontend;

class PrintFunctionNamesAction : public PluginParseTreeAction {

  // Visitor struct that defines Pre/Post functions for different types of nodes
  struct ParseTreeVisitor {
    template <typename A> bool Pre(const A &) { return true; }
    template <typename A> void Post(const A &) {}

    bool Pre(const Fortran::parser::FunctionSubprogram &) {
      isInSubprogram_ = true;
      return true;
    }
    void Post(const Fortran::parser::FunctionStmt &f) {
      if (isInSubprogram_) {
        llvm::outs() << "Function:\t"
                     << std::get<Fortran::parser::Name>(f.t).ToString() << "\n";
        fcounter++;
        isInSubprogram_ = false;
      }
    }

    bool Pre(const Fortran::parser::SubroutineSubprogram &) {
      isInSubprogram_ = true;
      return true;
    }
    void Post(const Fortran::parser::SubroutineStmt &s) {
      if (isInSubprogram_) {
        llvm::outs() << "Subroutine:\t"
                     << std::get<Fortran::parser::Name>(s.t).ToString() << "\n";
        scounter++;
        isInSubprogram_ = false;
      }
    }

    int fcounter{0};
    int scounter{0};

  private:
    bool isInSubprogram_{false};
  };

  void executeAction() override {
    ParseTreeVisitor visitor;
    Fortran::parser::Walk(getParsing().parseTree(), visitor);

    llvm::outs() << "\n====   Functions: " << visitor.fcounter << " ====\n";
    llvm::outs() << "==== Subroutines: " << visitor.scounter << " ====\n";
  }
};

static FrontendPluginRegistry::Add<PrintFunctionNamesAction> X(
    "print-fns", "Print Function names");
