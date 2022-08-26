#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/edit_distance.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>

using namespace llvm;
using namespace bolt;

FunctionFragment::FunctionFragment(FunctionLayout &Layout,
                                   const FragmentNum Num)
    : Layout(&Layout), Num(Num), StartIndex(Layout.block_size()) {}

FunctionFragment::iterator FunctionFragment::begin() {
  return iterator(Layout->block_begin() + StartIndex);
}
FunctionFragment::const_iterator FunctionFragment::begin() const {
  return const_iterator(Layout->block_begin() + StartIndex);
}
FunctionFragment::iterator FunctionFragment::end() {
  return iterator(Layout->block_begin() + StartIndex + Size);
}
FunctionFragment::const_iterator FunctionFragment::end() const {
  return const_iterator(Layout->block_begin() + StartIndex + Size);
}

const BinaryBasicBlock *FunctionFragment::front() const { return *begin(); }

FunctionLayout::FunctionLayout() { addFragment(); }

FunctionLayout::FunctionLayout(const FunctionLayout &Other)
    : Blocks(Other.Blocks) {
  for (FunctionFragment *const FF : Other.Fragments) {
    auto *Copy = new FunctionFragment(*FF);
    Copy->Layout = this;
    Fragments.emplace_back(Copy);
  }
}

FunctionLayout::FunctionLayout(FunctionLayout &&Other)
    : Fragments(std::move(Other.Fragments)), Blocks(std::move(Other.Blocks)) {
  for (FunctionFragment *const F : Fragments)
    F->Layout = this;
}

FunctionLayout &FunctionLayout::operator=(const FunctionLayout &Other) {
  Blocks = Other.Blocks;
  for (FunctionFragment *const FF : Other.Fragments) {
    auto *const Copy = new FunctionFragment(*FF);
    Copy->Layout = this;
    Fragments.emplace_back(Copy);
  }
  return *this;
}

FunctionLayout &FunctionLayout::operator=(FunctionLayout &&Other) {
  Fragments = std::move(Other.Fragments);
  Blocks = std::move(Other.Blocks);
  for (FunctionFragment *const FF : Fragments)
    FF->Layout = this;
  return *this;
}

FunctionLayout::~FunctionLayout() {
  for (FunctionFragment *const F : Fragments) {
    delete F;
  }
}

FunctionFragment &FunctionLayout::addFragment() {
  FunctionFragment *const FF =
      new FunctionFragment(*this, FragmentNum(Fragments.size()));
  Fragments.emplace_back(FF);
  return *FF;
}

FunctionFragment &FunctionLayout::getFragment(FragmentNum Num) {
  return *Fragments[Num.get()];
}

const FunctionFragment &FunctionLayout::getFragment(FragmentNum Num) const {
  return *Fragments[Num.get()];
}

const FunctionFragment &
FunctionLayout::findFragment(const BinaryBasicBlock *const BB) const {
  return getFragment(BB->getFragmentNum());
}

void FunctionLayout::addBasicBlock(BinaryBasicBlock *const BB) {
  BB->setLayoutIndex(Blocks.size());
  Blocks.emplace_back(BB);
  Fragments.back()->Size++;
}

void FunctionLayout::insertBasicBlocks(
    const BinaryBasicBlock *const InsertAfter,
    const ArrayRef<BinaryBasicBlock *> NewBlocks) {
  block_iterator InsertBeforePos = Blocks.begin();
  FragmentNum InsertFragmentNum = FragmentNum::main();
  unsigned LayoutIndex = 0;

  if (InsertAfter) {
    InsertBeforePos = std::next(findBasicBlockPos(InsertAfter));
    InsertFragmentNum = InsertAfter->getFragmentNum();
    LayoutIndex = InsertAfter->getLayoutIndex();
  }

  llvm::copy(NewBlocks, std::inserter(Blocks, InsertBeforePos));

  for (BinaryBasicBlock *const BB : NewBlocks) {
    BB->setFragmentNum(InsertFragmentNum);
    BB->setLayoutIndex(LayoutIndex++);
  }

  const fragment_iterator InsertFragment =
      fragment_begin() + InsertFragmentNum.get();
  InsertFragment->Size += NewBlocks.size();

  const fragment_iterator TailBegin = std::next(InsertFragment);
  auto const UpdateFragment = [&](FunctionFragment &FF) {
    FF.StartIndex += NewBlocks.size();
    for (BinaryBasicBlock *const BB : FF)
      BB->setLayoutIndex(LayoutIndex++);
  };
  std::for_each(TailBegin, fragment_end(), UpdateFragment);
}

void FunctionLayout::eraseBasicBlocks(
    const DenseSet<const BinaryBasicBlock *> ToErase) {
  const auto IsErased = [&](const BinaryBasicBlock *const BB) {
    return ToErase.contains(BB);
  };

  unsigned TotalErased = 0;
  for (FunctionFragment &FF : fragments()) {
    unsigned Erased = count_if(FF, IsErased);
    FF.Size -= Erased;
    FF.StartIndex -= TotalErased;
    TotalErased += Erased;
  }
  llvm::erase_if(Blocks, IsErased);

  // Remove empty fragments at the end
  const auto IsEmpty = [](const FunctionFragment *const FF) {
    return FF->empty();
  };
  const FragmentListType::iterator EmptyTailBegin =
      llvm::find_if_not(reverse(Fragments), IsEmpty).base();
  std::for_each(EmptyTailBegin, Fragments.end(),
                [](FunctionFragment *const FF) { delete FF; });
  Fragments.erase(EmptyTailBegin, Fragments.end());

  updateLayoutIndices();
}

void FunctionLayout::updateLayoutIndices() {
  unsigned BlockIndex = 0;
  for (FunctionFragment &FF : fragments()) {
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
        llvm::all_of(fragments(), [](const FunctionFragment &FF) {
          return llvm::all_of(FF, [&](const BinaryBasicBlock *const BB) {
            return FF.Num == BB->getFragmentNum();
          });
        });
    if (EqualPartitioning)
      return false;
  }

  clear();

  // Generate fragments
  for (BinaryBasicBlock *const BB : NewLayout) {
    FragmentNum Num = BB->getFragmentNum();

    assert(Num >= Fragments.back()->getFragmentNum() &&
           "Blocks must be arranged such that fragments are monotonically "
           "increasing.");

    // Add empty fragments if necessary
    while (Fragments.back()->getFragmentNum() < Num)
      addFragment();

    // Set the next fragment to point one past the current BB
    addBasicBlock(BB);
  }

  return true;
}

void FunctionLayout::clear() {
  Blocks = BasicBlockListType();
  // If the binary does not have relocations and is not split, the function will
  // be written to the output stream at its original file offset (see
  // `RewriteInstance::rewriteFile`). Hence, when the layout is cleared, retain
  // the main fragment, so that this information is not lost.
  std::for_each(Fragments.begin() + 1, Fragments.end(),
                [](FunctionFragment *const FF) { delete FF; });
  Fragments = FragmentListType{Fragments.front()};
  getMainFragment().Size = 0;
}

const BinaryBasicBlock *
FunctionLayout::getBasicBlockAfter(const BinaryBasicBlock *BB,
                                   bool IgnoreSplits) const {
  const block_const_iterator BBPos = find(blocks(), BB);
  if (BBPos == block_end())
    return nullptr;

  const block_const_iterator BlockAfter = std::next(BBPos);
  if (BlockAfter == block_end())
    return nullptr;

  if (!IgnoreSplits)
    if (BlockAfter == getFragment(BB->getFragmentNum()).end())
      return nullptr;

  return *BlockAfter;
}

bool FunctionLayout::isSplit() const {
  const unsigned NonEmptyFragCount = llvm::count_if(
      fragments(), [](const FunctionFragment &FF) { return !FF.empty(); });
  return NonEmptyFragCount >= 2;
}

uint64_t FunctionLayout::getEditDistance(
    const ArrayRef<const BinaryBasicBlock *> OldBlockOrder) const {
  return ComputeEditDistance<const BinaryBasicBlock *>(OldBlockOrder, Blocks);
}

FunctionLayout::block_const_iterator
FunctionLayout::findBasicBlockPos(const BinaryBasicBlock *BB) const {
  return block_const_iterator(find(Blocks, BB));
}

FunctionLayout::block_iterator
FunctionLayout::findBasicBlockPos(const BinaryBasicBlock *BB) {
  return find(Blocks, BB);
}
