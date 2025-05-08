//===----------------------------------------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Target.h"

using namespace lld;
using namespace lld::elf;

namespace {
class ParasolTargetInfo final : public TargetInfo {
public:
    ParasolTargetInfo();
    RelExpr getRelExpr(RelType type, const Symbol &s,
        const uint8_t *loc) const override;
    void relocate(uint8_t *loc, const Relocation &rel,
        uint64_t val) const override;
};
}

void ParasolTargetInfo::relocate(
    uint8_t *loc,
    const Relocation &rel,
    uint64_t val
) const {
    switch (rel.type) {
    default:
        llvm_unreachable("unknown relocation");
    }
}

RelExpr ParasolTargetInfo::getRelExpr(
    RelType type,
    const Symbol &s,
    const uint8_t *loc
) const {
    switch (type) {
    default:
        error(getErrorLocation(loc) + "unknown relocation (" + Twine(type) +
        ") against symbol " + toString(s));
        return R_NONE;
    }
}

ParasolTargetInfo::ParasolTargetInfo() {
    pltHeaderSize = 32;
    pltEntrySize = 16;
    ipltEntrySize = 16;
}

TargetInfo *elf::getParasolTargetInfo() {
    static ParasolTargetInfo t;

    return &t;
}
