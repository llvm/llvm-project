

#include "InterpHelpers.h"
#include "Descriptor.h"
#include "InterpBlock.h"
#include "Program.h"

namespace clang {
namespace interp {

void ensureArraySize(Program &P, const Pointer &Ptr, unsigned RequestedIndex) {
  if (!Ptr.isBlockPointer())
    return;

  assert(Ptr.getFieldDesc());
  assert(Ptr.getDeclDesc());
  if (!Ptr.getDeclDesc()->isArray())
    return;

  // No fillers for these.
  if (!Ptr.isStatic() || Ptr.isUnknownSizeArray() || Ptr.block()->isDynamic())
    return;

  assert(Ptr.getFieldDesc()->isArray());

  bool NeedsRealloc = RequestedIndex >= Ptr.getNumAllocatedElems() &&
                      RequestedIndex < Ptr.getCapacity();

  // llvm::errs() << "NeedsRealloc: " << NeedsRealloc << '\n';
  if (!NeedsRealloc)
    return;

  assert(Ptr.getFieldDesc()->hasArrayFiller());
  unsigned RequestedSize = RequestedIndex + 1;
  assert(RequestedSize <= Ptr.getNumElems());

  const Descriptor *D = Ptr.getFieldDesc();
  ArraySize NewArraySize = ArraySize::getNextSize(RequestedSize, D->Capacity);

  const Descriptor *NewDesc = nullptr;
  if (D->isPrimitiveArray()) {
    NewDesc = P.allocateDescriptor(D->Source, D->getPrimType(),
                                   Descriptor::GlobalMD, NewArraySize,
                                   D->IsConst, D->IsTemporary, D->IsMutable);
  } else if (D->isCompositeArray()) {
    NewDesc = P.allocateDescriptor(D->Source, D->SourceType, D->ElemDesc,
                                   Descriptor::GlobalMD, NewArraySize,
                                   D->IsConst, D->IsTemporary, D->IsMutable);
  } else {
    llvm_unreachable("Should be either a primitive or composite array");
  }

  assert(NewDesc);
  P.reallocGlobal(const_cast<Block *>(Ptr.block()), NewDesc);
}

} // namespace interp
} // namespace clang
