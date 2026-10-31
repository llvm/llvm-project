// Derive final kernel resource metadata from allocated XeMachine IR.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include <cstdint>
#include <limits>

namespace inter {
#define GEN_PASS_DEF_RESOURCEINFO
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

class ResourceInfo : public inter::impl::ResourceInfoBase<ResourceInfo> {
public:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    function->removeAttr(kGrfUsedAttrName);
    function->removeAttr(kBarrierCountAttrName);
    function->removeAttr(kHasGlobalAtomicsAttrName);
    function->removeAttr(kHasNoStatelessWriteAttrName);
    function->removeAttr(kHasDpasAttrName);
    if (function.isExternal())
      return;

    IntegerAttr grfCount =
        function->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
    if (!grfCount || grfCount.getInt() <= 0 ||
        grfCount.getInt() > std::numeric_limits<int32_t>::max()) {
      function.emitError("resource info requires a positive ")
          << kGrfCountAttrName << " function attribute that fits in i32";
      return signalPassFailure();
    }
    for (StringRef name : {kSlmSizeAttrName, kScratchSizeAttrName}) {
      IntegerAttr size = function->getAttrOfType<IntegerAttr>(name);
      if (size && size.getInt() < 0) {
        function.emitError("resource info requires nonnegative ")
            << name << " when present";
        return signalPassFailure();
      }
    }

    FailureOr<KernelResourceUsage> usage =
        analyzeKernelResources(function, grfCount.getInt());
    if (failed(usage))
      return signalPassFailure();

    Builder builder(function.getContext());
    function->setAttr(kGrfUsedAttrName,
                      builder.getI32IntegerAttr(usage->grfUsed));
    function->setAttr(kBarrierCountAttrName,
                      builder.getI32IntegerAttr(usage->barrierCount));
    function->setAttr(kHasGlobalAtomicsAttrName,
                      builder.getBoolAttr(usage->hasGlobalAtomics));
    function->setAttr(kHasNoStatelessWriteAttrName,
                      builder.getBoolAttr(!usage->hasStatelessWrite));
    function->setAttr(kHasDpasAttrName, builder.getBoolAttr(usage->hasDpas));
  }
};

} // namespace
