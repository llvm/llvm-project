#ifndef BOLT_PASSES_INSERT_NEGATE_RA_STATE_PASS
#define BOLT_PASSES_INSERT_NEGATE_RA_STATE_PASS

#include "bolt/Passes/BinaryPasses.h"
#include <stack>

namespace llvm {
namespace bolt {

class InsertNegateRAState : public BinaryFunctionPass {
public:
  explicit InsertNegateRAState() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "insert-negate-ra-state-pass"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);
  bool addNegateRAStateAfterPacOrAuth(BinaryFunction &BF);
  void fixUnknownStates(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm
#endif
