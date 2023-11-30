#ifndef FORTRAN_LOWER_OPENMPMIXIN_H
#define FORTRAN_LOWER_OPENMPMIXIN_H

#include "FirMixin.h"

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
}

template <typename ConverterT>
class OpenMPMixin : public FirMixinBase<ConverterT> {
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
  // Shortcuts
  Fortran::lower::LoweringBridge &getBridge() {
    return this->This()->getBridge();
  }
  fir::FirOpBuilder &getBuilder() { return this->This()->getBuilder(); }
  Fortran::lower::pft::Evaluation &getEval() { return this->This()->getEval(); }
  Fortran::lower::SymMap &getSymTable() { return this->This()->getSymTable(); }

private:
  /// Whether a target region or declare target function/subroutine
  /// intended for device offloading have been detected
  bool ompDeviceCodeFound = false;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_OPENMPMIXIN_H
