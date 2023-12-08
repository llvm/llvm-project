//===-- OpenMPMixin.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMPMIXIN_H
#define FORTRAN_LOWER_OPENMPMIXIN_H

#include "ConverterMixin.h"
#include "flang/Parser/parse-tree.h"

namespace fir {
class FirOpBuilder;
}

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::lower {

class AbstractConverter;
class LoweringBridge;
class SymMap;

namespace pft {
class Evaluation;
class Variable;
} // namespace pft

template <typename ConverterT>
class OpenMPMixin : public ConverterMixinBase<ConverterT> {
public:
  void genFIR(const Fortran::parser::OpenMPConstruct &);
  void genFIR(const Fortran::parser::OpenMPDeclarativeConstruct &);

  void genFIR(const Fortran::parser::OmpEndLoopDirective &) {} // nop

  void instantiateVariable(Fortran::lower::AbstractConverter &converter,
                           const Fortran::lower::pft::Variable &var);
  void finalize(const Fortran::semantics::Symbol *globalOmpRequiresSymbol);

private:
  // Shortcuts to call ConverterT:: functions. They can't be defined here
  // because the definition of ConverterT is not available at this point.
  Fortran::lower::LoweringBridge &getBridge();
  fir::FirOpBuilder &getBuilder();
  Fortran::lower::pft::Evaluation &getEval();
  Fortran::lower::SymMap &getSymTable();

private:
  /// Whether a target region or declare target function/subroutine
  /// intended for device offloading have been detected
  bool ompDeviceCodeFound = false;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_OPENMPMIXIN_H
