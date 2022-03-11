#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"

namespace clang {
namespace dataflow {

/// Dataflow analysis that discovers unsafe accesses of optional values and
/// adds the respective source locations to the lattice.
///
/// Models the `std::optional`, `absl::optional`, and `base::Optional` types.
///
/// FIXME: Consider separating the models from the unchecked access analysis.
class UncheckedOptionalAccessModel
    : public DataflowAnalysis<UncheckedOptionalAccessModel,
                              SourceLocationsLattice> {
public:
  explicit UncheckedOptionalAccessModel(ASTContext &AstContext);

  static SourceLocationsLattice initialElement() {
    return SourceLocationsLattice();
  }

  void transfer(const Stmt *Stmt, SourceLocationsLattice &State,
                Environment &Env);

private:
  MatchSwitch<TransferState<SourceLocationsLattice>> TransferMatchSwitch;
};

} // namespace dataflow
} // namespace clang

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDOPTIONALACCESSMODEL_H
