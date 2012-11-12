//===--- Lambda.h - Types for C++ Lambdas -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief  Defines several types used to describe C++ lambda expressions
/// that are shared between the parser and AST.
///
//===----------------------------------------------------------------------===//


#ifndef LLVM_CLANG_BASIC_LAMBDA_H
#define LLVM_CLANG_BASIC_LAMBDA_H

namespace clang {

/// \brief The default, if any, capture method for a lambda expression.
enum LambdaCaptureDefault {
  LCD_None,
  LCD_ByCopy,
  LCD_ByRef
};

/// \brief The different capture forms in a lambda introducer: 'this' or a
/// copied or referenced variable.
enum LambdaCaptureKind {
  LCK_This,
  LCK_ByCopy,
  LCK_ByRef
};

} // end namespace clang

#endif // LLVM_CLANG_BASIC_LAMBDA_H
