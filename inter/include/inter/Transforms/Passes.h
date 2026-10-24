#ifndef INTER_TRANSFORMS_PASSES_H
#define INTER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace inter {

#define GEN_PASS_DECL
#include "inter/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "inter/Transforms/Passes.h.inc"

} // namespace inter

#endif // INTER_TRANSFORMS_PASSES_H
