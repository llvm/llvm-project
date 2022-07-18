#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/edit_distance.h"
#include <algorithm>
#include <functional>

using namespace llvm;
using namespace bolt;

unsigned FunctionFragment::size() const { return end() - begin(); }
bool FunctionFragment::empty() const { return end() == begin(); }
FunctionFragment::const_iterator FunctionFragment::begin() const {
  return Layout.block_begin() + Layout.Fragments[Num.get()];
}
FunctionFragment::const_iterator FunctionFragment::end() const {
  return Layout.block_begin() + Layout.Fragments[Num.get() + 1];
}
BinaryBasicBlock *FunctionFragment::front() const { return *begin(); }

FunctionFragment FunctionLayout::addFragment() {
  Fragments.emplace_back(Blocks.size());
  return back();
}

FunctionFragment FunctionLayout::getFragment(FragmentNum Num) const {
  return FunctionFragment(Num, *this);
}

FunctionFragment
FunctionLayout::findFragment(const BinaryBasicBlock *BB) const {
  return getFragment(BB->getFragmentNum());
}

void FunctionLayout::addBasicBlock(BinaryBasicBlock *BB) {
  if (empty())
    addFragment();

  BB->setLayoutIndex(Blocks.size());
  Blocks.emplace_back(BB);
  ++Fragments.back();

  assert(Fragments.back() == Blocks.size());
}

void FunctionLayout::insertBasicBlocks(BinaryBasicBlock *InsertAfter,
                                       ArrayRef<BinaryBasicBlock *> NewBlocks) {
  if (empty())
    addFragment();

  const block_iterator InsertBeforePos =
      InsertAfter ? std::next(findBasicBlockPos(InsertAfter)) : Blocks.begin();
  Blocks.insert(InsertBeforePos, NewBlocks.begin(), NewBlocks.end());

  unsigned FragmentUpdateStart =
      InsertAfter ? InsertAfter->getFragmentNum().get() + 1 : 1;
  std::for_each(
      Fragments.begin() + FragmentUpdateStart, Fragments.end(),
      [&](unsigned &FragmentOffset) { FragmentOffset += NewBlocks.size(); });
}

void FunctionLayout::eraseBasicBlocks(
    const DenseSet<const BinaryBasicBlock *> ToErase) {
  auto IsErased = [&](const BinaryBasicBlock *const BB) {
    return ToErase.contains(BB);
  };
  FragmentListType NewFragments;
  NewFragments.emplace_back(0);
  for (const FunctionFragment F : *this) {
    unsigned ErasedBlocks = count_if(F, IsErased);
    unsigned NewFragment = NewFragments.back() + F.size() - ErasedBlocks;
    NewFragments.emplace_back(NewFragment);
  }
  Blocks.erase(std::remove_if(Blocks.begin(), Blocks.end(), IsErased),
               Blocks.end());
  Fragments = std::move(NewFragments);
}

void FunctionLayout::updateLayoutIndices() const {
  unsigned BlockIndex = 0;
  for (const FunctionFragment F : *this) {
    for (BinaryBasicBlock *const BB : F)
      BB->setLayoutIndex(BlockIndex++);
  }
}

void FunctionLayout::update(const ArrayRef<BinaryBasicBlock *> NewLayout) {
  PreviousBlocks = std::move(Blocks);
  PreviousFragments = std::move(Fragments);

  Blocks.clear();
  Fragments = {0};

  if (NewLayout.empty())
    return;

  copy(NewLayout, std::back_inserter(Blocks));

  // Generate fragments
  for (const auto &BB : enumerate(Blocks)) {
    unsigned FragmentNum = BB.value()->getFragmentNum().get();

    assert(FragmentNum + 1 >= size() &&
           "Blocks must be arranged such that fragments are monotonically "
           "increasing.");

    // Add empty fragments if necessary
    for (unsigned I = size(); I <= FragmentNum; ++I) {
      addFragment();
      Fragments[I] = BB.index();
    }

    // Set the next fragment to point one past the current BB
    Fragments[FragmentNum + 1] = BB.index() + 1;
  }

  if (PreviousBlocks == Blocks && PreviousFragments == Fragments) {
    // If new layout is the same as previous layout, clear previous layout so
    // hasLayoutChanged() returns false.
    PreviousBlocks = {};
    PreviousFragments = {};
  }
}

void FunctionLayout::clear() {
  Blocks = {};
  Fragments = {0};
  PreviousBlocks = {};
  PreviousFragments = {0};
}

BinaryBasicBlock *FunctionLayout::getBasicBlockAfter(const BinaryBasicBlock *BB,
                                                     bool IgnoreSplits) const {
  const block_const_iterator BBPos = find(Blocks, BB);
  if (BBPos == Blocks.end())
    return nullptr;

  const block_const_iterator BlockAfter = std::next(BBPos);
  if (BlockAfter == Blocks.end())
    return nullptr;

  if (!IgnoreSplits)
    if (BlockAfter == getFragment(BB->getFragmentNum()).end())
      return nullptr;

  return *BlockAfter;
}

bool FunctionLayout::isSplit() const {
  unsigned NonEmptyFragCount = llvm::count_if(
      *this, [](const FunctionFragment &F) { return !F.empty(); });
  return NonEmptyFragCount >= 2;
}

uint64_t FunctionLayout::getEditDistance() const {
  return ComputeEditDistance<BinaryBasicBlock *>(PreviousBlocks, Blocks);
}

FunctionLayout::block_const_iterator
FunctionLayout::findBasicBlockPos(const BinaryBasicBlock *BB) const {
  return find(Blocks, BB);
}

FunctionLayout::block_iterator
FunctionLayout::findBasicBlockPos(const BinaryBasicBlock *BB) {
  return find(Blocks, BB);
}
