//===- LifetimeAnnotations.h -  -*--------------- C++--------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Helper functions to inspect and infer lifetime annotations.
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMEANNOTATIONS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMEANNOTATIONS_H

#include "clang/AST/DeclCXX.h"

namespace clang ::lifetimes {

// This function is needed because Decl::isInStdNamespace will return false for
// iterators in some STL implementations due to them being defined in a
// namespace outside of the std namespace.
bool isInStlNamespace(const Decl *D);

bool isPointerLikeType(QualType QT);

/// Returns the most recent declaration of the method to ensure all
/// lifetime-bound attributes from redeclarations are considered.
const FunctionDecl *getDeclWithMergedLifetimeBoundAttrs(const FunctionDecl *FD);

/// Returns the most recent declaration of the method to ensure all
/// lifetime-bound attributes from redeclarations are considered.
const CXXMethodDecl *
getDeclWithMergedLifetimeBoundAttrs(const CXXMethodDecl *CMD);

// Return true if this is an "normal" assignment operator.
// We assume that a normal assignment operator always returns *this, that is,
// an lvalue reference that is the same type as the implicit object parameter
// (or the LHS for a non-member operator==).
bool isNormalAssignmentOperator(const FunctionDecl *FD);

/// Returns true if this is an assignment operator where the parameter
/// has the lifetimebound attribute.
bool isAssignmentOperatorLifetimeBound(const CXXMethodDecl *CMD);

/// Returns true if the implicit object parameter (this) should be considered
/// lifetimebound, either due to an explicit lifetimebound attribute on the
/// method or because it's a normal assignment operator.
bool implicitObjectParamIsLifetimeBound(const FunctionDecl *FD);

// Returns true if the implicit object argument (this) of a method call should
// be tracked for GSL lifetime analysis. This applies to STL methods that return
// pointers or references that depend on the lifetime of the object, such as
// container iterators (begin, end), data accessors (c_str, data, get), or
// element accessors (operator[], operator*, front, back, at).
bool shouldTrackImplicitObjectArg(const CXXMethodDecl *Callee);

// Returns true if the first argument of a free function should be tracked for
// GSL lifetime analysis. This applies to STL free functions that take a pointer
// to a GSL Owner or Pointer and return a pointer or reference that depends on
// the lifetime of the argument, such as std::begin, std::data, std::get, or
// std::any_cast.
bool shouldTrackFirstArgument(const FunctionDecl *FD);

// Tells whether the type is annotated with [[gsl::Pointer]].
bool isGslPointerType(QualType QT);
// Tells whether the type is annotated with [[gsl::Owner]].
bool isGslOwnerType(QualType QT);

} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMEANNOTATIONS_H
