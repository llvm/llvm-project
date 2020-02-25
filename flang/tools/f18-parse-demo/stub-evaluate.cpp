//===-- tools/f18/stub-evaluate.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The parse tree has slots in which pointers to the results of semantic
// analysis may be placed.  When using the parser without the semantics
// libraries, as here, we need to stub out the dependences on the external
// destructors, which will never actually be called.

#include "flang/Common/indirection.h"

namespace Fortran::evaluate {
struct GenericExprWrapper {
  ~GenericExprWrapper();
};
GenericExprWrapper::~GenericExprWrapper() {}
struct GenericAssignmentWrapper {
  ~GenericAssignmentWrapper();
};
GenericAssignmentWrapper::~GenericAssignmentWrapper() {}
struct ProcedureRef {
  ~ProcedureRef();
};
ProcedureRef::~ProcedureRef() {}
}

DEFINE_DELETER(Fortran::evaluate::GenericExprWrapper)
DEFINE_DELETER(Fortran::evaluate::GenericAssignmentWrapper)
DEFINE_DELETER(Fortran::evaluate::ProcedureRef)
