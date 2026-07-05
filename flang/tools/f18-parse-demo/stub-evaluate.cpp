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
// deleters, which will never actually be called.

namespace Fortran::evaluate {
struct GenericExprWrapper {
  static void Deleter(GenericExprWrapper *);
};
void GenericExprWrapper::Deleter(GenericExprWrapper *) {}
struct GenericAssignmentWrapper {
  static void Deleter(GenericAssignmentWrapper *);
};
void GenericAssignmentWrapper::Deleter(GenericAssignmentWrapper *) {}
struct ProcedureRef {
  static void Deleter(ProcedureRef *);
};
void ProcedureRef::Deleter(ProcedureRef *) {}
} // namespace Fortran::evaluate
