//===- CallDescription.cpp - function/method call matching     --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines a generic mechanism for matching for function and
/// method calls of C, C++, and Objective-C languages. Instances of these
/// classes are frequently used together with the CallEvent classes.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

using namespace llvm;
using namespace clang;

// A constructor helper.
static Optional<size_t> readRequiredParams(Optional<unsigned> RequiredArgs,
                                           Optional<size_t> RequiredParams) {
  if (RequiredParams)
    return RequiredParams;
  if (RequiredArgs)
    return static_cast<size_t>(*RequiredArgs);
  return None;
}

ento::CallDescription::CallDescription(
    int Flags, ArrayRef<const char *> QualifiedName,
    Optional<unsigned> RequiredArgs /*= None*/,
    Optional<size_t> RequiredParams /*= None*/)
    : QualifiedName(QualifiedName), RequiredArgs(RequiredArgs),
      RequiredParams(readRequiredParams(RequiredArgs, RequiredParams)),
      Flags(Flags) {
  assert(!QualifiedName.empty());
}

/// Construct a CallDescription with default flags.
ento::CallDescription::CallDescription(
    ArrayRef<const char *> QualifiedName,
    Optional<unsigned> RequiredArgs /*= None*/,
    Optional<size_t> RequiredParams /*= None*/)
    : CallDescription(0, QualifiedName, RequiredArgs, RequiredParams) {}
