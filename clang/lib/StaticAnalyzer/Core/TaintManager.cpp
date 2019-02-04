//== TaintManager.cpp ------------------------------------------ -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/TaintManager.h"

using namespace clang;
using namespace ento;

void *ProgramStateTrait<TaintMap>::GDMIndex() {
  static int index = 0;
  return &index;
}

void *ProgramStateTrait<DerivedSymTaint>::GDMIndex() {
  static int index;
  return &index;
}
