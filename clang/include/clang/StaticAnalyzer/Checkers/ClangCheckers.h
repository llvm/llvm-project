//===--- ClangCheckers.h - Provides builtin checkers ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CHECKERS_CLANGCHECKERS_H
#define LLVM_CLANG_STATICANALYZER_CHECKERS_CLANGCHECKERS_H

namespace clang {
namespace ento {
class CheckerRegistry;

void registerBuiltinCheckers(CheckerRegistry &registry);

} // end namespace ento
} // end namespace clang

#endif
