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
  bool BBhasAUTH(BinaryContext &BC, BinaryBasicBlock *BB);
  bool BBhasSIGN(BinaryContext &BC, BinaryBasicBlock *BB);
  void explore_call_graph(BinaryContext &BC, BinaryBasicBlock *BB);
  void process_signed_BB(BinaryContext &BC, BinaryBasicBlock *BB,
                         std::stack<BinaryBasicBlock *> *SignedStack,
                         std::stack<BinaryBasicBlock *> *UnsignedStack);
  void process_unsigned_BB(BinaryContext &BC, BinaryBasicBlock *BB,
                           std::stack<BinaryBasicBlock *> *SignedStack,
                           std::stack<BinaryBasicBlock *> *UnsignedStack);
};

} // namespace bolt
} // namespace llvm
#endif
