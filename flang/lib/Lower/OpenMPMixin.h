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
class SemanticsContext;
class Symbol;
} // namespace Fortran::semantics

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
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPSimpleStandaloneConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPFlushConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPCancelConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPCancellationPointConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPStandaloneConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPSectionsConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPSectionConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPLoopConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPBlockConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPAtomicConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPCriticalConstruct &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPExecutableAllocate &);

  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPAllocatorsConstruct &);

  // Declarative
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPDeclarativeAllocate &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPDeclareReductionConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPDeclareSimdConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPDeclareTargetConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPRequiresConstruct &);
  void genOMP(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::pft::Evaluation &eval,
              Fortran::semantics::SemanticsContext &semaCtx,
              const Fortran::parser::OpenMPThreadprivate &);

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
