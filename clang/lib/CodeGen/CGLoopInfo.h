//===---- CGLoopInfo.h - LLVM CodeGen for loop metadata -*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal state used for llvm translation for loop statement
// metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGLOOPINFO_H
#define LLVM_CLANG_LIB_CODEGEN_CGLOOPINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class BasicBlock;
class Instruction;
class MDNode;
} // end namespace llvm

namespace clang {
class Attr;
class ASTContext;
namespace CodeGen {

/// \brief Attributes that may be specified on loops.
struct LoopAttributes {
  explicit LoopAttributes(bool IsParallel = false);
  void clear();

  /// \brief Generate llvm.loop.parallel metadata for loads and stores.
  bool IsParallel;

  /// \brief State of loop vectorization or unrolling.
  enum LVEnableState { Unspecified, Enable, Disable, Full };

  /// \brief Value for llvm.loop.vectorize.enable metadata.
  LVEnableState VectorizeEnable;

  /// \brief Value for llvm.loop.unroll.* metadata (enable, disable, or full).
  LVEnableState UnrollEnable;

  /// \brief Value for llvm.loop.vectorize.width metadata.
  unsigned VectorizeWidth;

  /// \brief Value for llvm.loop.interleave.count metadata.
  unsigned InterleaveCount;

  /// \brief llvm.unroll.
  unsigned UnrollCount;

  /// \brief Value for llvm.loop.distribute.enable metadata.
  LVEnableState DistributeEnable;
};

/// \brief Information used when generating a structured loop.
class LoopInfo {
public:
  /// \brief Construct a new LoopInfo for the loop with entry Header.
  LoopInfo(llvm::BasicBlock *Header, const LoopAttributes &Attrs,
           llvm::DebugLoc Location);

  /// \brief Get the loop id metadata for this loop.
  llvm::MDNode *getLoopID() const { return LoopID; }

  /// \brief Get the header block of this loop.
  llvm::BasicBlock *getHeader() const { return Header; }

  /// \brief Get the set of attributes active for this loop.
  const LoopAttributes &getAttributes() const { return Attrs; }

private:
  /// \brief Loop ID metadata.
  llvm::MDNode *LoopID;
  /// \brief Header block of this loop.
  llvm::BasicBlock *Header;
  /// \brief The attributes for this loop.
  LoopAttributes Attrs;
};

/// \brief A stack of loop information corresponding to loop nesting levels.
/// This stack can be used to prepare attributes which are applied when a loop
/// is emitted.
class LoopInfoStack {
  LoopInfoStack(const LoopInfoStack &) = delete;
  void operator=(const LoopInfoStack &) = delete;

public:
  LoopInfoStack() {}

  /// \brief Begin a new structured loop. The set of staged attributes will be
  /// applied to the loop and then cleared.
  void push(llvm::BasicBlock *Header,
            llvm::DebugLoc Location = llvm::DebugLoc());

  /// \brief Begin a new structured loop. Stage attributes from the Attrs list.
  /// The staged attributes are applied to the loop and then cleared.
  void push(llvm::BasicBlock *Header, clang::ASTContext &Ctx,
            llvm::ArrayRef<const Attr *> Attrs,
            llvm::DebugLoc Location = llvm::DebugLoc());

  /// \brief End the current loop.
  void pop();

  /// \brief Return the top loop id metadata.
  llvm::MDNode *getCurLoopID() const { return getInfo().getLoopID(); }

  /// \brief Return true if the top loop is parallel.
  bool getCurLoopParallel() const {
    return hasInfo() ? getInfo().getAttributes().IsParallel : false;
  }

  /// \brief Function called by the CodeGenFunction when an instruction is
  /// created.
  void InsertHelper(llvm::Instruction *I) const;

  /// \brief Set the next pushed loop as parallel.
  void setParallel(bool Enable = true) { StagedAttrs.IsParallel = Enable; }

  /// \brief Set the next pushed loop 'vectorize.enable'
  void setVectorizeEnable(bool Enable = true) {
    StagedAttrs.VectorizeEnable =
        Enable ? LoopAttributes::Enable : LoopAttributes::Disable;
  }

  /// \brief Set the next pushed loop as a distribution candidate.
  void setDistributeState(bool Enable = true) {
    StagedAttrs.DistributeEnable =
        Enable ? LoopAttributes::Enable : LoopAttributes::Disable;
  }

  /// \brief Set the next pushed loop unroll state.
  void setUnrollState(const LoopAttributes::LVEnableState &State) {
    StagedAttrs.UnrollEnable = State;
  }

  /// \brief Set the vectorize width for the next loop pushed.
  void setVectorizeWidth(unsigned W) { StagedAttrs.VectorizeWidth = W; }

  /// \brief Set the interleave count for the next loop pushed.
  void setInterleaveCount(unsigned C) { StagedAttrs.InterleaveCount = C; }

  /// \brief Set the unroll count for the next loop pushed.
  void setUnrollCount(unsigned C) { StagedAttrs.UnrollCount = C; }

private:
  /// \brief Returns true if there is LoopInfo on the stack.
  bool hasInfo() const { return !Active.empty(); }
  /// \brief Return the LoopInfo for the current loop. HasInfo should be called
  /// first to ensure LoopInfo is present.
  const LoopInfo &getInfo() const { return Active.back(); }
  /// \brief The set of attributes that will be applied to the next pushed loop.
  LoopAttributes StagedAttrs;
  /// \brief Stack of active loops.
  llvm::SmallVector<LoopInfo, 4> Active;
};

} // end namespace CodeGen
} // end namespace clang

#endif
