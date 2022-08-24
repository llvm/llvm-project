#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/edit_distance.h"
#include <algorithm>
#include <cstddef>
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
  return getFragment(FragmentNum(Blocks.size() - 1));
}

FunctionFragment FunctionLayout::getFragment(FragmentNum Num) const {
  return FunctionFragment(Num, *this);
}

FunctionFragment
FunctionLayout::findFragment(const BinaryBasicBlock *BB) const {
  return getFragment(BB->getFragmentNum());
}

void FunctionLayout::addBasicBlock(BinaryBasicBlock *BB) {
  BB->setLayoutIndex(Blocks.size());
  Blocks.emplace_back(BB);
  ++Fragments.back();
  assert(Fragments.back() == Blocks.size());
}

void FunctionLayout::insertBasicBlocks(BinaryBasicBlock *InsertAfter,
                                       ArrayRef<BinaryBasicBlock *> NewBlocks) {
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
  for (const FunctionFragment FF : fragments()) {
    unsigned ErasedBlocks = count_if(FF, IsErased);
    // Only add the fragment if it is non-empty after removing blocks.
    unsigned NewFragment = NewFragments.back() + FF.size() - ErasedBlocks;
    NewFragments.emplace_back(NewFragment);
  }
  llvm::erase_if(Blocks, IsErased);
  Fragments = std::move(NewFragments);

  // Remove empty fragments at the end
  const_iterator EmptyTailBegin =
      llvm::find_if_not(reverse(fragments()), [](const FunctionFragment &FF) {
        return FF.empty();
      }).base();
  if (EmptyTailBegin != fragment_end()) {
    // Add +1 for one-past-the-end entry
    const FunctionFragment TailBegin = *EmptyTailBegin;
    unsigned NewFragmentSize = TailBegin.getFragmentNum().get() + 1;
    Fragments.resize(NewFragmentSize);
  }

  updateLayoutIndices();
}

void FunctionLayout::updateLayoutIndices() const {
  unsigned BlockIndex = 0;
  for (const FunctionFragment FF : fragments()) {
    for (BinaryBasicBlock *const BB : FF) {
      BB->setLayoutIndex(BlockIndex++);
      BB->setFragmentNum(FF.getFragmentNum());
    }
  }
}

bool FunctionLayout::update(const ArrayRef<BinaryBasicBlock *> NewLayout) {
  const bool EqualBlockOrder = llvm::equal(Blocks, NewLayout);
  if (EqualBlockOrder) {
    const bool EqualPartitioning =
        llvm::all_of(fragments(), [](const FunctionFragment FF) {
          return llvm::all_of(FF, [&](const BinaryBasicBlock *const BB) {
            return FF.Num == BB->getFragmentNum();
          });
        });
    if (EqualPartitioning)
      return false;
  }

  Blocks = BasicBlockListType(NewLayout.begin(), NewLayout.end());
  Fragments = {0, 0};

  // Generate fragments
  for (const auto &BB : enumerate(Blocks)) {
    unsigned FragmentNum = BB.value()->getFragmentNum().get();

    assert(FragmentNum >= fragment_size() - 1 &&
           "Blocks must be arranged such that fragments are monotonically "
           "increasing.");

    // Add empty fragments if necessary
    for (unsigned I = fragment_size(); I <= FragmentNum; ++I) {
      addFragment();
      Fragments[I] = BB.index();
    }

    // Set the next fragment to point one past the current BB
    Fragments[FragmentNum + 1] = BB.index() + 1;
  }

  return true;
}

void FunctionLayout::clear() {
  Blocks = {};
  Fragments = {0, 0};
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
      fragments(), [](const FunctionFragment &FF) { return !FF.empty(); });
  return NonEmptyFragCount >= 2;
}

uint64_t FunctionLayout::getEditDistance(
    const ArrayRef<const BinaryBasicBlock *> OldBlockOrder) const {
  return ComputeEditDistance<const BinaryBasicBlock *>(OldBlockOrder, Blocks);
}

FunctionLayout::block_const_iterator
FunctionLayout::findBasicBlockPos(const BinaryBasicBlock *BB) const {
  return find(Blocks, BB);
}

FunctionLayout::block_iterator
FunctionLayout::findBasicBlockPos(const BinaryBasicBlock *BB) {
  return find(Blocks, BB);
}
