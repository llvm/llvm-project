
#include "Pointer.h"

namespace clang {
namespace interp {
void ensureArraySize(Program &P, const Pointer &Ptr, unsigned RequestedIndex);

inline void ensureArraySize(Program &P, const Pointer &Ptr) {
  if (!Ptr.getFieldDesc()->isArray())
    return;
  ensureArraySize(P, Ptr, Ptr.getNumElems() - 1);
}
} // namespace interp
} // namespace clang
