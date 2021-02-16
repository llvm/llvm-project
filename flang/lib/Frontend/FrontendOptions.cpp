//===- FrontendOptions.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendOptions.h"
#include "flang/Evaluate/expression.h"

using namespace Fortran::frontend;

bool Fortran::frontend::isFixedFormSuffix(llvm::StringRef suffix) {
  // Note: Keep this list in-sync with flang/test/lit.cfg.py
  return suffix == "f" || suffix == "F" || suffix == "ff" || suffix == "for" ||
      suffix == "FOR" || suffix == "fpp" || suffix == "FPP";
}

bool Fortran::frontend::isFreeFormSuffix(llvm::StringRef suffix) {
  // Note: Keep this list in-sync with flang/test/lit.cfg.py
  // TODO: Add Cuda Fortan files (i.e. `*.cuf` and `*.CUF`).
  return suffix == "f77" || suffix == "f90" || suffix == "F90" ||
      suffix == "ff90" || suffix == "f95" || suffix == "F95" ||
      suffix == "ff95" || suffix == "f03" || suffix == "F03" ||
      suffix == "f08" || suffix == "F08" || suffix == "f18" || suffix == "F18";
}

// TODO: This is a copy of `asFortran` from f18.cpp and is added here for
// compatiblity. It doesn't really belong here, but I couldn't find a better
// place. We should decide whether to add it to the Evaluate or Parse/Unparse
// APIs or some dedicated utility library in the driver.
Fortran::parser::AnalyzedObjectsAsFortran
Fortran::frontend::getBasicAsFortran() {
  return Fortran::parser::AnalyzedObjectsAsFortran{
      [](llvm::raw_ostream &o, const Fortran::evaluate::GenericExprWrapper &x) {
        if (x.v) {
          x.v->AsFortran(o);
        } else {
          o << "(bad expression)";
        }
      },
      [](llvm::raw_ostream &o,
          const Fortran::evaluate::GenericAssignmentWrapper &x) {
        if (x.v) {
          x.v->AsFortran(o);
        } else {
          o << "(bad assignment)";
        }
      },
      [](llvm::raw_ostream &o, const Fortran::evaluate::ProcedureRef &x) {
        x.AsFortran(o << "CALL ");
      },
  };
}

InputKind FrontendOptions::GetInputKindForExtension(llvm::StringRef extension) {
  if (isFixedFormSuffix(extension) || isFreeFormSuffix(extension)) {
    return Language::Fortran;
  }
  return Language::Unknown;
}
