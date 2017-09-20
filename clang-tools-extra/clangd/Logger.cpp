//===--- Logger.cpp - Logger interface for clangd -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Logger.h"

using namespace clang::clangd;

EmptyLogger &EmptyLogger::getInstance() {
  static EmptyLogger Logger;
  return Logger;
}

void EmptyLogger::log(const llvm::Twine &Message) {}
