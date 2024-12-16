#ifndef BOLT_PASSES_MARK_RA_STATES
#define BOLT_PASSES_MARK_RA_STATES

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

class MarkRAStates : public BinaryFunctionPass {
public:
  explicit MarkRAStates() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "mark-ra-states"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
  void runOnFunction(BinaryFunction &BF);
};

} // namespace bolt
} // namespace llvm
#endif
