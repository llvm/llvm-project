//===-- HelloWorldPlugin.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic example Flang plugin which simply prints a Hello World statement
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"

using namespace Fortran::frontend;

class HelloWorldFlangPlugin : public PluginParseTreeAction {
  void ExecuteAction() override {
    llvm::outs() << "Hello World from your new Flang plugin\n";
  }
};

static FrontendPluginRegistry::Add<HelloWorldFlangPlugin> X(
    "-hello-world", "Hello World Plugin example");
