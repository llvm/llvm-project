//== Checker.cpp - Registration mechanism for checkers -----------*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines Checker, used to create and register checkers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/Checker.h"

using namespace clang;
using namespace ento;

int ImplicitNullDerefEvent::Tag;

StringRef CheckerBase::getDebugTag() const { return getName(); }

void CheckerBackend::printState(raw_ostream &Out, ProgramStateRef State,
                                const char *NL, const char *Sep) const {}
