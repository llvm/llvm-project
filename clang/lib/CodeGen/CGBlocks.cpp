//===--- CGBlocks.cpp - Emit LLVM Code for declarations -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit blocks.
//
//===----------------------------------------------------------------------===//

#include "CGBlocks.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <cstdio>

using namespace clang;
using namespace CodeGen;

CGBlockInfo::CGBlockInfo(const BlockDecl *block, StringRef name)
  : Name(name), CXXThisIndex(0), CanBeGlobal(false), NeedsCopyDispose(false),
    HasCXXObject(false), UsesStret(false), HasCapturedVariableLayout(false),
    StructureType(nullptr), Block(block),
    DominatingIP(nullptr) {

  // Skip asm prefix, if any.  'name' is usually taken directly from
  // the mangled name of the enclosing function.
  if (!name.empty() && name[0] == '\01')
    name = name.substr(1);
}

// Anchor the vtable to this translation unit.
CodeGenModule::ByrefHelpers::~ByrefHelpers() {}

/// Build the given block as a global block.
static llvm::Constant *buildGlobalBlock(CodeGenModule &CGM,
                                        const CGBlockInfo &blockInfo,
                                        llvm::Constant *blockFn);

/// Build the helper function to copy a block.
static llvm::Constant *buildCopyHelper(CodeGenModule &CGM,
                                       const CGBlockInfo &blockInfo) {
  return CodeGenFunction(CGM).GenerateCopyHelperFunction(blockInfo);
}

/// Build the helper function to dispose of a block.
static llvm::Constant *buildDisposeHelper(CodeGenModule &CGM,
                                          const CGBlockInfo &blockInfo) {
  return CodeGenFunction(CGM).GenerateDestroyHelperFunction(blockInfo);
}

/// buildBlockDescriptor - Build the block descriptor meta-data for a block.
/// buildBlockDescriptor is accessed from 5th field of the Block_literal
/// meta-data and contains stationary information about the block literal.
/// Its definition will have 4 (or optinally 6) words.
/// \code
/// struct Block_descriptor {
///   unsigned long reserved;
///   unsigned long size;  // size of Block_literal metadata in bytes.
///   void *copy_func_helper_decl;  // optional copy helper.
///   void *destroy_func_decl; // optioanl destructor helper.
///   void *block_method_encoding_address; // @encode for block literal signature.
///   void *block_layout_info; // encoding of captured block variables.
/// };
/// \endcode
static llvm::Constant *buildBlockDescriptor(CodeGenModule &CGM,
                                            const CGBlockInfo &blockInfo) {
  ASTContext &C = CGM.getContext();

  llvm::Type *ulong = CGM.getTypes().ConvertType(C.UnsignedLongTy);
  llvm::Type *i8p = NULL;
  if (CGM.getLangOpts().OpenCL)
    i8p = 
      llvm::Type::getInt8PtrTy(
           CGM.getLLVMContext(), C.getTargetAddressSpace(LangAS::opencl_constant));
  else
    i8p = CGM.getTypes().ConvertType(C.VoidPtrTy);

  SmallVector<llvm::Constant*, 6> elements;

  // reserved
  elements.push_back(llvm::ConstantInt::get(ulong, 0));

  // Size
  // FIXME: What is the right way to say this doesn't fit?  We should give
  // a user diagnostic in that case.  Better fix would be to change the
  // API to size_t.
  elements.push_back(llvm::ConstantInt::get(ulong,
                                            blockInfo.BlockSize.getQuantity()));

  // Optional copy/dispose helpers.
  if (blockInfo.NeedsCopyDispose) {
    // copy_func_helper_decl
    elements.push_back(buildCopyHelper(CGM, blockInfo));

    // destroy_func_decl
    elements.push_back(buildDisposeHelper(CGM, blockInfo));
  }

  // Signature.  Mandatory ObjC-style method descriptor @encode sequence.
  std::string typeAtEncoding =
    CGM.getContext().getObjCEncodingForBlock(blockInfo.getBlockExpr());
  elements.push_back(llvm::ConstantExpr::getBitCast(
                          CGM.GetAddrOfConstantCString(typeAtEncoding), i8p));
  
  // GC layout.
  if (C.getLangOpts().ObjC1) {
    if (CGM.getLangOpts().getGC() != LangOptions::NonGC)
      elements.push_back(CGM.getObjCRuntime().BuildGCBlockLayout(CGM, blockInfo));
    else
      elements.push_back(CGM.getObjCRuntime().BuildRCBlockLayout(CGM, blockInfo));
  }
  else
    elements.push_back(llvm::Constant::getNullValue(i8p));

  llvm::Constant *init = llvm::ConstantStruct::getAnon(elements);

  llvm::GlobalVariable *global =
    new llvm::GlobalVariable(CGM.getModule(), init->getType(), true,
                             llvm::GlobalValue::InternalLinkage,
                             init, "__block_descriptor_tmp");

  return llvm::ConstantExpr::getBitCast(global, CGM.getBlockDescriptorType());
}

/*
  Purely notional variadic template describing the layout of a block.

  template <class _ResultType, class... _ParamTypes, class... _CaptureTypes>
  struct Block_literal {
    /// Initialized to one of:
    ///   extern void *_NSConcreteStackBlock[];
    ///   extern void *_NSConcreteGlobalBlock[];
    ///
    /// In theory, we could start one off malloc'ed by setting
    /// BLOCK_NEEDS_FREE, giving it a refcount of 1, and using
    /// this isa:
    ///   extern void *_NSConcreteMallocBlock[];
    struct objc_class *isa;

    /// These are the flags (with corresponding bit number) that the
    /// compiler is actually supposed to know about.
    ///  25. BLOCK_HAS_COPY_DISPOSE - indicates that the block
    ///   descriptor provides copy and dispose helper functions
    ///  26. BLOCK_HAS_CXX_OBJ - indicates that there's a captured
    ///   object with a nontrivial destructor or copy constructor
    ///  28. BLOCK_IS_GLOBAL - indicates that the block is allocated
    ///   as global memory
    ///  29. BLOCK_USE_STRET - indicates that the block function
    ///   uses stret, which objc_msgSend needs to know about
    ///  30. BLOCK_HAS_SIGNATURE - indicates that the block has an
    ///   @encoded signature string
    /// And we're not supposed to manipulate these:
    ///  24. BLOCK_NEEDS_FREE - indicates that the block has been moved
    ///   to malloc'ed memory
    ///  27. BLOCK_IS_GC - indicates that the block has been moved to
    ///   to GC-allocated memory
    /// Additionally, the bottom 16 bits are a reference count which
    /// should be zero on the stack.
    int flags;

    /// Reserved;  should be zero-initialized.
    int reserved;

    /// Function pointer generated from block literal.
    _ResultType (*invoke)(Block_literal *, _ParamTypes...);

    /// Block description metadata generated from block literal.
    struct Block_descriptor *block_descriptor;

    /// Captured values follow.
    _CapturesTypes captures...;
  };
 */

/// The number of fields in a block header.
const unsigned BlockHeaderSize = 5;

namespace {
  /// A chunk of data that we actually have to capture in the block.
  struct BlockLayoutChunk {
    CharUnits Alignment;
    CharUnits Size;
    Qualifiers::ObjCLifetime Lifetime;
    const BlockDecl::Capture *Capture; // null for 'this'
    llvm::Type *Type;

    BlockLayoutChunk(CharUnits align, CharUnits size,
                     Qualifiers::ObjCLifetime lifetime,
                     const BlockDecl::Capture *capture,
                     llvm::Type *type)
      : Alignment(align), Size(size), Lifetime(lifetime),
        Capture(capture), Type(type) {}

    /// Tell the block info that this chunk has the given field index.
    void setIndex(CGBlockInfo &info, unsigned index) {
      if (!Capture)
        info.CXXThisIndex = index;
      else
        info.Captures[Capture->getVariable()]
          = CGBlockInfo::Capture::makeIndex(index);
    }
  };

  /// Order by 1) all __strong together 2) next, all byfref together 3) next,
  /// all __weak together. Preserve descending alignment in all situations.
  bool operator<(const BlockLayoutChunk &left, const BlockLayoutChunk &right) {
    CharUnits LeftValue, RightValue;
    bool LeftByref = left.Capture ? left.Capture->isByRef() : false;
    bool RightByref = right.Capture ? right.Capture->isByRef() : false;
    
    if (left.Lifetime == Qualifiers::OCL_Strong &&
        left.Alignment >= right.Alignment)
      LeftValue = CharUnits::fromQuantity(64);
    else if (LeftByref && left.Alignment >= right.Alignment)
      LeftValue = CharUnits::fromQuantity(32);
    else if (left.Lifetime == Qualifiers::OCL_Weak &&
             left.Alignment >= right.Alignment)
      LeftValue = CharUnits::fromQuantity(16);
    else
      LeftValue = left.Alignment;
    if (right.Lifetime == Qualifiers::OCL_Strong &&
        right.Alignment >= left.Alignment)
      RightValue = CharUnits::fromQuantity(64);
    else if (RightByref && right.Alignment >= left.Alignment)
      RightValue = CharUnits::fromQuantity(32);
    else if (right.Lifetime == Qualifiers::OCL_Weak &&
             right.Alignment >= left.Alignment)
      RightValue = CharUnits::fromQuantity(16);
    else
      RightValue = right.Alignment;
    
      return LeftValue > RightValue;
  }
}

/// Determines if the given type is safe for constant capture in C++.
static bool isSafeForCXXConstantCapture(QualType type) {
  const RecordType *recordType =
    type->getBaseElementTypeUnsafe()->getAs<RecordType>();

  // Only records can be unsafe.
  if (!recordType) return true;

  const auto *record = cast<CXXRecordDecl>(recordType->getDecl());

  // Maintain semantics for classes with non-trivial dtors or copy ctors.
  if (!record->hasTrivialDestructor()) return false;
  if (record->hasNonTrivialCopyConstructor()) return false;

  // Otherwise, we just have to make sure there aren't any mutable
  // fields that might have changed since initialization.
  return !record->hasMutableFields();
}

/// It is illegal to modify a const object after initialization.
/// Therefore, if a const object has a constant initializer, we don't
/// actually need to keep storage for it in the block; we'll just
/// rematerialize it at the start of the block function.  This is
/// acceptable because we make no promises about address stability of
/// captured variables.
static llvm::Constant *tryCaptureAsConstant(CodeGenModule &CGM,
                                            CodeGenFunction *CGF,
                                            const VarDecl *var) {
  QualType type = var->getType();

  // We can only do this if the variable is const.
  if (!type.isConstQualified()) return nullptr;

  // Furthermore, in C++ we have to worry about mutable fields:
  // C++ [dcl.type.cv]p4:
  //   Except that any class member declared mutable can be
  //   modified, any attempt to modify a const object during its
  //   lifetime results in undefined behavior.
  if (CGM.getLangOpts().CPlusPlus && !isSafeForCXXConstantCapture(type))
    return nullptr;

  // If the variable doesn't have any initializer (shouldn't this be
  // invalid?), it's not clear what we should do.  Maybe capture as
  // zero?
  const Expr *init = var->getInit();
  if (!init) return nullptr;

  return CGM.EmitConstantInit(*var, CGF);
}

/// Get the low bit of a nonzero character count.  This is the
/// alignment of the nth byte if the 0th byte is universally aligned.
static CharUnits getLowBit(CharUnits v) {
  return CharUnits::fromQuantity(v.getQuantity() & (~v.getQuantity() + 1));
}

static void initializeForBlockHeader(CodeGenModule &CGM, CGBlockInfo &info,
                             SmallVectorImpl<llvm::Type*> &elementTypes) {
  ASTContext &C = CGM.getContext();

  // The header is basically a 'struct { void *; int; int; void *; void *; }'.
  CharUnits ptrSize, ptrAlign, intSize, intAlign;
  std::tie(ptrSize, ptrAlign) = C.getTypeInfoInChars(C.VoidPtrTy);
  std::tie(intSize, intAlign) = C.getTypeInfoInChars(C.IntTy);

  // Are there crazy embedded platforms where this isn't true?
  assert(intSize <= ptrSize && "layout assumptions horribly violated");

  CharUnits headerSize = ptrSize;
  if (2 * intSize < ptrAlign) headerSize += ptrSize;
  else headerSize += 2 * intSize;
  headerSize += 2 * ptrSize;

  info.BlockAlign = ptrAlign;
  info.BlockSize = headerSize;

  assert(elementTypes.empty());
  llvm::Type *i8p = CGM.getTypes().ConvertType(C.VoidPtrTy);
  llvm::Type *intTy = CGM.getTypes().ConvertType(C.IntTy);
  elementTypes.push_back(i8p);
  elementTypes.push_back(intTy);
  elementTypes.push_back(intTy);
  elementTypes.push_back(i8p);
  elementTypes.push_back(CGM.getBlockDescriptorType());

  assert(elementTypes.size() == BlockHeaderSize);
}

/// Compute the layout of the given block.  Attempts to lay the block
/// out with minimal space requirements.
static void computeBlockInfo(CodeGenModule &CGM, CodeGenFunction *CGF,
                             CGBlockInfo &info) {
  ASTContext &C = CGM.getContext();
  const BlockDecl *block = info.getBlockDecl();

  SmallVector<llvm::Type*, 8> elementTypes;
  initializeForBlockHeader(CGM, info, elementTypes);

  if (!block->hasCaptures()) {
    info.StructureType =
      llvm::StructType::get(CGM.getLLVMContext(), elementTypes, true);
    info.CanBeGlobal = true;
    return;
  }
  else if (C.getLangOpts().ObjC1 &&
           CGM.getLangOpts().getGC() == LangOptions::NonGC)
    info.HasCapturedVariableLayout = true;
  
  // Collect the layout chunks.
  SmallVector<BlockLayoutChunk, 16> layout;
  layout.reserve(block->capturesCXXThis() +
                 (block->capture_end() - block->capture_begin()));

  CharUnits maxFieldAlign;

  // First, 'this'.
  if (block->capturesCXXThis()) {
    assert(CGF && CGF->CurFuncDecl && isa<CXXMethodDecl>(CGF->CurFuncDecl) &&
           "Can't capture 'this' outside a method");
    QualType thisType = cast<CXXMethodDecl>(CGF->CurFuncDecl)->getThisType(C);

    llvm::Type *llvmType = CGM.getTypes().ConvertType(thisType);
    std::pair<CharUnits,CharUnits> tinfo
      = CGM.getContext().getTypeInfoInChars(thisType);
    maxFieldAlign = std::max(maxFieldAlign, tinfo.second);

    layout.push_back(BlockLayoutChunk(tinfo.second, tinfo.first,
                                      Qualifiers::OCL_None,
                                      nullptr, llvmType));
  }

  // Next, all the block captures.
  for (const auto &CI : block->captures()) {
    const VarDecl *variable = CI.getVariable();

    if (CI.isByRef()) {
      // We have to copy/dispose of the __block reference.
      info.NeedsCopyDispose = true;

      // Just use void* instead of a pointer to the byref type.
      QualType byRefPtrTy = C.VoidPtrTy;

      llvm::Type *llvmType = CGM.getTypes().ConvertType(byRefPtrTy);
      std::pair<CharUnits,CharUnits> tinfo
        = CGM.getContext().getTypeInfoInChars(byRefPtrTy);
      maxFieldAlign = std::max(maxFieldAlign, tinfo.second);

      layout.push_back(BlockLayoutChunk(tinfo.second, tinfo.first,
                                        Qualifiers::OCL_None, &CI, llvmType));
      continue;
    }

    // Otherwise, build a layout chunk with the size and alignment of
    // the declaration.
    if (llvm::Constant *constant = tryCaptureAsConstant(CGM, CGF, variable)) {
      info.Captures[variable] = CGBlockInfo::Capture::makeConstant(constant);
      continue;
    }

    // If we have a lifetime qualifier, honor it for capture purposes.
    // That includes *not* copying it if it's __unsafe_unretained.
    Qualifiers::ObjCLifetime lifetime =
      variable->getType().getObjCLifetime();
    if (lifetime) {
      switch (lifetime) {
      case Qualifiers::OCL_None: llvm_unreachable("impossible");
      case Qualifiers::OCL_ExplicitNone:
      case Qualifiers::OCL_Autoreleasing:
        break;

      case Qualifiers::OCL_Strong:
      case Qualifiers::OCL_Weak:
        info.NeedsCopyDispose = true;
      }

    // Block pointers require copy/dispose.  So do Objective-C pointers.
    } else if (variable->getType()->isObjCRetainableType()) {
      info.NeedsCopyDispose = true;
      // used for mrr below.
      lifetime = Qualifiers::OCL_Strong;

    // So do types that require non-trivial copy construction.
    } else if (CI.hasCopyExpr()) {
      info.NeedsCopyDispose = true;
      info.HasCXXObject = true;

    // And so do types with destructors.
    } else if (CGM.getLangOpts().CPlusPlus) {
      if (const CXXRecordDecl *record =
            variable->getType()->getAsCXXRecordDecl()) {
        if (!record->hasTrivialDestructor()) {
          info.HasCXXObject = true;
          info.NeedsCopyDispose = true;
        }
      }
    }

    QualType VT = variable->getType();
    CharUnits size = C.getTypeSizeInChars(VT);
    CharUnits align = C.getDeclAlign(variable);
    
    maxFieldAlign = std::max(maxFieldAlign, align);

    llvm::Type *llvmType =
      CGM.getTypes().ConvertTypeForMem(VT);
    
    layout.push_back(BlockLayoutChunk(align, size, lifetime, &CI, llvmType));
  }

  // If that was everything, we're done here.
  if (layout.empty()) {
    info.StructureType =
      llvm::StructType::get(CGM.getLLVMContext(), elementTypes, true);
    info.CanBeGlobal = true;
    return;
  }

  // Sort the layout by alignment.  We have to use a stable sort here
  // to get reproducible results.  There should probably be an
  // llvm::array_pod_stable_sort.
  std::stable_sort(layout.begin(), layout.end());
  
  // Needed for blocks layout info.
  info.BlockHeaderForcedGapOffset = info.BlockSize;
  info.BlockHeaderForcedGapSize = CharUnits::Zero();
  
  CharUnits &blockSize = info.BlockSize;
  info.BlockAlign = std::max(maxFieldAlign, info.BlockAlign);

  // Assuming that the first byte in the header is maximally aligned,
  // get the alignment of the first byte following the header.
  CharUnits endAlign = getLowBit(blockSize);

  // If the end of the header isn't satisfactorily aligned for the
  // maximum thing, look for things that are okay with the header-end
  // alignment, and keep appending them until we get something that's
  // aligned right.  This algorithm is only guaranteed optimal if
  // that condition is satisfied at some point; otherwise we can get
  // things like:
  //   header                 // next byte has alignment 4
  //   something_with_size_5; // next byte has alignment 1
  //   something_with_alignment_8;
  // which has 7 bytes of padding, as opposed to the naive solution
  // which might have less (?).
  if (endAlign < maxFieldAlign) {
    SmallVectorImpl<BlockLayoutChunk>::iterator
      li = layout.begin() + 1, le = layout.end();

    // Look for something that the header end is already
    // satisfactorily aligned for.
    for (; li != le && endAlign < li->Alignment; ++li)
      ;

    // If we found something that's naturally aligned for the end of
    // the header, keep adding things...
    if (li != le) {
      SmallVectorImpl<BlockLayoutChunk>::iterator first = li;
      for (; li != le; ++li) {
        assert(endAlign >= li->Alignment);

        li->setIndex(info, elementTypes.size());
        elementTypes.push_back(li->Type);
        blockSize += li->Size;
        endAlign = getLowBit(blockSize);

        // ...until we get to the alignment of the maximum field.
        if (endAlign >= maxFieldAlign) {
          if (li == first) {
            // No user field was appended. So, a gap was added.
            // Save total gap size for use in block layout bit map.
            info.BlockHeaderForcedGapSize = li->Size;
          }
          break;
        }
      }
      // Don't re-append everything we just appended.
      layout.erase(first, li);
    }
  }

  assert(endAlign == getLowBit(blockSize));
  
  // At this point, we just have to add padding if the end align still
  // isn't aligned right.
  if (endAlign < maxFieldAlign) {
    CharUnits newBlockSize = blockSize.RoundUpToAlignment(maxFieldAlign);
    CharUnits padding = newBlockSize - blockSize;

    elementTypes.push_back(llvm::ArrayType::get(CGM.Int8Ty,
                                                padding.getQuantity()));
    blockSize = newBlockSize;
    endAlign = getLowBit(blockSize); // might be > maxFieldAlign
  }

  assert(endAlign >= maxFieldAlign);
  assert(endAlign == getLowBit(blockSize));
  // Slam everything else on now.  This works because they have
  // strictly decreasing alignment and we expect that size is always a
  // multiple of alignment.
  for (SmallVectorImpl<BlockLayoutChunk>::iterator
         li = layout.begin(), le = layout.end(); li != le; ++li) {
    assert(endAlign >= li->Alignment);
    li->setIndex(info, elementTypes.size());
    elementTypes.push_back(li->Type);
    blockSize += li->Size;
    endAlign = getLowBit(blockSize);
  }

  info.StructureType =
    llvm::StructType::get(CGM.getLLVMContext(), elementTypes, true);
}

/// Enter the scope of a block.  This should be run at the entrance to
/// a full-expression so that the block's cleanups are pushed at the
/// right place in the stack.
static void enterBlockScope(CodeGenFunction &CGF, BlockDecl *block) {
  assert(CGF.HaveInsertPoint());

  // Allocate the block info and place it at the head of the list.
  CGBlockInfo &blockInfo =
    *new CGBlockInfo(block, CGF.CurFn->getName());
  blockInfo.NextBlockInfo = CGF.FirstBlockInfo;
  CGF.FirstBlockInfo = &blockInfo;

  // Compute information about the layout, etc., of this block,
  // pushing cleanups as necessary.
  computeBlockInfo(CGF.CGM, &CGF, blockInfo);

  // Nothing else to do if it can be global.
  if (blockInfo.CanBeGlobal) return;

  // Make the allocation for the block.
  blockInfo.Address =
    CGF.CreateTempAlloca(blockInfo.StructureType, "block");
  blockInfo.Address->setAlignment(blockInfo.BlockAlign.getQuantity());

  // If there are cleanups to emit, enter them (but inactive).
  if (!blockInfo.NeedsCopyDispose) return;

  // Walk through the captures (in order) and find the ones not
  // captured by constant.
  for (const auto &CI : block->captures()) {
    // Ignore __block captures; there's nothing special in the
    // on-stack block that we need to do for them.
    if (CI.isByRef()) continue;

    // Ignore variables that are constant-captured.
    const VarDecl *variable = CI.getVariable();
    CGBlockInfo::Capture &capture = blockInfo.getCapture(variable);
    if (capture.isConstant()) continue;

    // Ignore objects that aren't destructed.
    QualType::DestructionKind dtorKind =
      variable->getType().isDestructedType();
    if (dtorKind == QualType::DK_none) continue;

    CodeGenFunction::Destroyer *destroyer;

    // Block captures count as local values and have imprecise semantics.
    // They also can't be arrays, so need to worry about that.
    if (dtorKind == QualType::DK_objc_strong_lifetime) {
      destroyer = CodeGenFunction::destroyARCStrongImprecise;
    } else {
      destroyer = CGF.getDestroyer(dtorKind);
    }

    // GEP down to the address.
    llvm::Value *addr = CGF.Builder.CreateStructGEP(blockInfo.Address,
                                                    capture.getIndex());

    // We can use that GEP as the dominating IP.
    if (!blockInfo.DominatingIP)
      blockInfo.DominatingIP = cast<llvm::Instruction>(addr);

    CleanupKind cleanupKind = InactiveNormalCleanup;
    bool useArrayEHCleanup = CGF.needsEHCleanup(dtorKind);
    if (useArrayEHCleanup) 
      cleanupKind = InactiveNormalAndEHCleanup;

    CGF.pushDestroy(cleanupKind, addr, variable->getType(),
                    destroyer, useArrayEHCleanup);

    // Remember where that cleanup was.
    capture.setCleanup(CGF.EHStack.stable_begin());
  }
}

/// Enter a full-expression with a non-trivial number of objects to
/// clean up.  This is in this file because, at the moment, the only
/// kind of cleanup object is a BlockDecl*.
void CodeGenFunction::enterNonTrivialFullExpression(const ExprWithCleanups *E) {
  assert(E->getNumObjects() != 0);
  ArrayRef<ExprWithCleanups::CleanupObject> cleanups = E->getObjects();
  for (ArrayRef<ExprWithCleanups::CleanupObject>::iterator
         i = cleanups.begin(), e = cleanups.end(); i != e; ++i) {
    enterBlockScope(*this, *i);
  }
}

/// Find the layout for the given block in a linked list and remove it.
static CGBlockInfo *findAndRemoveBlockInfo(CGBlockInfo **head,
                                           const BlockDecl *block) {
  while (true) {
    assert(head && *head);
    CGBlockInfo *cur = *head;

    // If this is the block we're looking for, splice it out of the list.
    if (cur->getBlockDecl() == block) {
      *head = cur->NextBlockInfo;
      return cur;
    }

    head = &cur->NextBlockInfo;
  }
}

/// Destroy a chain of block layouts.
void CodeGenFunction::destroyBlockInfos(CGBlockInfo *head) {
  assert(head && "destroying an empty chain");
  do {
    CGBlockInfo *cur = head;
    head = cur->NextBlockInfo;
    delete cur;
  } while (head != nullptr);
}

/// Emit a block literal expression in the current function.
llvm::Value *CodeGenFunction::EmitBlockLiteral(const BlockExpr *blockExpr) {
  // If the block has no captures, we won't have a pre-computed
  // layout for it.
  if (!blockExpr->getBlockDecl()->hasCaptures()) {
    CGBlockInfo blockInfo(blockExpr->getBlockDecl(), CurFn->getName());
    computeBlockInfo(CGM, this, blockInfo);
    blockInfo.BlockExpression = blockExpr;
    return EmitBlockLiteral(blockInfo);
  }

  // Find the block info for this block and take ownership of it.
  std::unique_ptr<CGBlockInfo> blockInfo;
  blockInfo.reset(findAndRemoveBlockInfo(&FirstBlockInfo,
                                         blockExpr->getBlockDecl()));

  blockInfo->BlockExpression = blockExpr;
  return EmitBlockLiteral(*blockInfo);
}

llvm::Value *CodeGenFunction::EmitBlockLiteral(const CGBlockInfo &blockInfo) {
  // Using the computed layout, generate the actual block function.
  bool isLambdaConv = blockInfo.getBlockDecl()->isConversionFromLambda();
  llvm::Constant *blockFn
    = CodeGenFunction(CGM, true).GenerateBlockFunction(CurGD, blockInfo,
                                                       LocalDeclMap,
                                                       isLambdaConv);
  blockFn = llvm::ConstantExpr::getBitCast(blockFn, VoidPtrTy);

  // If there is nothing to capture, we can emit this as a global block.
  if (blockInfo.CanBeGlobal)
    return buildGlobalBlock(CGM, blockInfo, blockFn);

  // Otherwise, we have to emit this as a local block.

  llvm::Constant *isa = CGM.getNSConcreteStackBlock();
  isa = llvm::ConstantExpr::getBitCast(isa, VoidPtrTy);

  // Build the block descriptor.
  llvm::Constant *descriptor = buildBlockDescriptor(CGM, blockInfo);

  llvm::AllocaInst *blockAddr = blockInfo.Address;
  assert(blockAddr && "block has no address!");

  // Compute the initial on-stack block flags.
  BlockFlags flags = BLOCK_HAS_SIGNATURE;
  if (blockInfo.HasCapturedVariableLayout) flags |= BLOCK_HAS_EXTENDED_LAYOUT;
  if (blockInfo.NeedsCopyDispose) flags |= BLOCK_HAS_COPY_DISPOSE;
  if (blockInfo.HasCXXObject) flags |= BLOCK_HAS_CXX_OBJ;
  if (blockInfo.UsesStret) flags |= BLOCK_USE_STRET;

  // Initialize the block literal.
  Builder.CreateStore(isa, Builder.CreateStructGEP(blockAddr, 0, "block.isa"));
  Builder.CreateStore(llvm::ConstantInt::get(IntTy, flags.getBitMask()),
                      Builder.CreateStructGEP(blockAddr, 1, "block.flags"));
  Builder.CreateStore(llvm::ConstantInt::get(IntTy, 0),
                      Builder.CreateStructGEP(blockAddr, 2, "block.reserved"));
  Builder.CreateStore(blockFn, Builder.CreateStructGEP(blockAddr, 3,
                                                       "block.invoke"));
  Builder.CreateStore(descriptor, Builder.CreateStructGEP(blockAddr, 4,
                                                          "block.descriptor"));

  // Finally, capture all the values into the block.
  const BlockDecl *blockDecl = blockInfo.getBlockDecl();

  // First, 'this'.
  if (blockDecl->capturesCXXThis()) {
    llvm::Value *addr = Builder.CreateStructGEP(blockAddr,
                                                blockInfo.CXXThisIndex,
                                                "block.captured-this.addr");
    Builder.CreateStore(LoadCXXThis(), addr);
  }

  // Next, captured variables.
  for (const auto &CI : blockDecl->captures()) {
    const VarDecl *variable = CI.getVariable();
    const CGBlockInfo::Capture &capture = blockInfo.getCapture(variable);

    // Ignore constant captures.
    if (capture.isConstant()) continue;

    QualType type = variable->getType();
    CharUnits align = getContext().getDeclAlign(variable);

    // This will be a [[type]]*, except that a byref entry will just be
    // an i8**.
    llvm::Value *blockField =
      Builder.CreateStructGEP(blockAddr, capture.getIndex(),
                              "block.captured");

    // Compute the address of the thing we're going to move into the
    // block literal.
    llvm::Value *src;
    if (BlockInfo && CI.isNested()) {
      // We need to use the capture from the enclosing block.
      const CGBlockInfo::Capture &enclosingCapture =
        BlockInfo->getCapture(variable);

      // This is a [[type]]*, except that a byref entry wil just be an i8**.
      src = Builder.CreateStructGEP(LoadBlockStruct(),
                                    enclosingCapture.getIndex(),
                                    "block.capture.addr");
    } else if (blockDecl->isConversionFromLambda()) {
      // The lambda capture in a lambda's conversion-to-block-pointer is
      // special; we'll simply emit it directly.
      src = nullptr;
    } else {
      // Just look it up in the locals map, which will give us back a
      // [[type]]*.  If that doesn't work, do the more elaborate DRE
      // emission.
      src = LocalDeclMap.lookup(variable);
      if (!src) {
        DeclRefExpr declRef(const_cast<VarDecl *>(variable),
                            /*refersToEnclosing*/ CI.isNested(), type,
                            VK_LValue, SourceLocation());
        src = EmitDeclRefLValue(&declRef).getAddress();
      }
    }

    // For byrefs, we just write the pointer to the byref struct into
    // the block field.  There's no need to chase the forwarding
    // pointer at this point, since we're building something that will
    // live a shorter life than the stack byref anyway.
    if (CI.isByRef()) {
      // Get a void* that points to the byref struct.
      if (CI.isNested())
        src = Builder.CreateAlignedLoad(src, align.getQuantity(),
                                        "byref.capture");
      else
        src = Builder.CreateBitCast(src, VoidPtrTy);

      // Write that void* into the capture field.
      Builder.CreateAlignedStore(src, blockField, align.getQuantity());

    // If we have a copy constructor, evaluate that into the block field.
    } else if (const Expr *copyExpr = CI.getCopyExpr()) {
      if (blockDecl->isConversionFromLambda()) {
        // If we have a lambda conversion, emit the expression
        // directly into the block instead.
        AggValueSlot Slot =
            AggValueSlot::forAddr(blockField, align, Qualifiers(),
                                  AggValueSlot::IsDestructed,
                                  AggValueSlot::DoesNotNeedGCBarriers,
                                  AggValueSlot::IsNotAliased);
        EmitAggExpr(copyExpr, Slot);
      } else {
        EmitSynthesizedCXXCopyCtor(blockField, src, copyExpr);
      }

    // If it's a reference variable, copy the reference into the block field.
    } else if (type->isReferenceType()) {
      llvm::Value *ref =
        Builder.CreateAlignedLoad(src, align.getQuantity(), "ref.val");
      Builder.CreateAlignedStore(ref, blockField, align.getQuantity());

    // If this is an ARC __strong block-pointer variable, don't do a
    // block copy.
    //
    // TODO: this can be generalized into the normal initialization logic:
    // we should never need to do a block-copy when initializing a local
    // variable, because the local variable's lifetime should be strictly
    // contained within the stack block's.
    } else if (type.getObjCLifetime() == Qualifiers::OCL_Strong &&
               type->isBlockPointerType()) {
      // Load the block and do a simple retain.
      LValue srcLV = MakeAddrLValue(src, type, align);
      llvm::Value *value = EmitLoadOfScalar(srcLV, SourceLocation());
      value = EmitARCRetainNonBlock(value);

      // Do a primitive store to the block field.
      LValue destLV = MakeAddrLValue(blockField, type, align);
      EmitStoreOfScalar(value, destLV, /*init*/ true);

    // Otherwise, fake up a POD copy into the block field.
    } else {
      // Fake up a new variable so that EmitScalarInit doesn't think
      // we're referring to the variable in its own initializer.
      ImplicitParamDecl blockFieldPseudoVar(getContext(), /*DC*/ nullptr,
                                            SourceLocation(), /*name*/ nullptr,
                                            type);

      // We use one of these or the other depending on whether the
      // reference is nested.
      DeclRefExpr declRef(const_cast<VarDecl*>(variable),
                          /*refersToEnclosing*/ CI.isNested(), type,
                          VK_LValue, SourceLocation());

      ImplicitCastExpr l2r(ImplicitCastExpr::OnStack, type, CK_LValueToRValue,
                           &declRef, VK_RValue);
      EmitExprAsInit(&l2r, &blockFieldPseudoVar,
                     MakeAddrLValue(blockField, type, align),
                     /*captured by init*/ false);
    }

    // Activate the cleanup if layout pushed one.
    if (!CI.isByRef()) {
      EHScopeStack::stable_iterator cleanup = capture.getCleanup();
      if (cleanup.isValid())
        ActivateCleanupBlock(cleanup, blockInfo.DominatingIP);
    }
  }

  // Cast to the converted block-pointer type, which happens (somewhat
  // unfortunately) to be a pointer to function type.
  llvm::Value *result =
    Builder.CreateBitCast(blockAddr,
                          ConvertType(blockInfo.getBlockExpr()->getType()));

  return result;
}


llvm::Type *CodeGenModule::getBlockDescriptorType() {
  if (BlockDescriptorType)
    return BlockDescriptorType;

  llvm::Type *UnsignedLongTy =
    getTypes().ConvertType(getContext().UnsignedLongTy);

  // struct __block_descriptor {
  //   unsigned long reserved;
  //   unsigned long block_size;
  //
  //   // later, the following will be added
  //
  //   struct {
  //     void (*copyHelper)();
  //     void (*copyHelper)();
  //   } helpers;                // !!! optional
  //
  //   const char *signature;   // the block signature
  //   const char *layout;      // reserved
  // };
  BlockDescriptorType =
    llvm::StructType::create("struct.__block_descriptor",
                             UnsignedLongTy, UnsignedLongTy, NULL);

  // Now form a pointer to that.
  BlockDescriptorType = llvm::PointerType::getUnqual(BlockDescriptorType);
  return BlockDescriptorType;
}

llvm::Type *CodeGenModule::getGenericBlockLiteralType() {
  if (GenericBlockLiteralType)
    return GenericBlockLiteralType;

  llvm::Type *BlockDescPtrTy = getBlockDescriptorType();

  // struct __block_literal_generic {
  //   void *__isa;
  //   int __flags;
  //   int __reserved;
  //   void (*__invoke)(void *);
  //   struct __block_descriptor *__descriptor;
  // };
  GenericBlockLiteralType =
    llvm::StructType::create("struct.__block_literal_generic",
                             VoidPtrTy, IntTy, IntTy, VoidPtrTy,
                             BlockDescPtrTy, NULL);

  return GenericBlockLiteralType;
}


RValue CodeGenFunction::EmitBlockCallExpr(const CallExpr *E, 
                                          ReturnValueSlot ReturnValue) {
  const BlockPointerType *BPT =
    E->getCallee()->getType()->getAs<BlockPointerType>();

  llvm::Value *Callee = EmitScalarExpr(E->getCallee());

  // Get a pointer to the generic block literal.
  llvm::Type *BlockLiteralTy =
    llvm::PointerType::getUnqual(CGM.getGenericBlockLiteralType());

  // Bitcast the callee to a block literal.
  llvm::Value *BlockLiteral =
    Builder.CreateBitCast(Callee, BlockLiteralTy, "block.literal");

  // Get the function pointer from the literal.
  llvm::Value *FuncPtr = Builder.CreateStructGEP(BlockLiteral, 3);

  BlockLiteral = Builder.CreateBitCast(BlockLiteral, VoidPtrTy);

  // Add the block literal.
  CallArgList Args;
  Args.add(RValue::get(BlockLiteral), getContext().VoidPtrTy);

  QualType FnType = BPT->getPointeeType();

  // And the rest of the arguments.
  EmitCallArgs(Args, FnType->getAs<FunctionProtoType>(),
               E->arg_begin(), E->arg_end());

  // Load the function.
  llvm::Value *Func = Builder.CreateLoad(FuncPtr);

  const FunctionType *FuncTy = FnType->castAs<FunctionType>();
  const CGFunctionInfo &FnInfo =
    CGM.getTypes().arrangeBlockFunctionCall(Args, FuncTy);

  // Cast the function pointer to the right type.
  llvm::Type *BlockFTy = CGM.getTypes().GetFunctionType(FnInfo);

  llvm::Type *BlockFTyPtr = llvm::PointerType::getUnqual(BlockFTy);
  Func = Builder.CreateBitCast(Func, BlockFTyPtr);

  // And call the block.
  return EmitCall(FnInfo, Func, ReturnValue, Args);
}

llvm::Value *CodeGenFunction::GetAddrOfBlockDecl(const VarDecl *variable,
                                                 bool isByRef) {
  assert(BlockInfo && "evaluating block ref without block information?");
  const CGBlockInfo::Capture &capture = BlockInfo->getCapture(variable);

  // Handle constant captures.
  if (capture.isConstant()) return LocalDeclMap[variable];

  llvm::Value *addr =
    Builder.CreateStructGEP(LoadBlockStruct(), capture.getIndex(),
                            "block.capture.addr");

  if (isByRef) {
    // addr should be a void** right now.  Load, then cast the result
    // to byref*.

    addr = Builder.CreateLoad(addr);
    llvm::PointerType *byrefPointerType
      = llvm::PointerType::get(BuildByRefType(variable), 0);
    addr = Builder.CreateBitCast(addr, byrefPointerType,
                                 "byref.addr");

    // Follow the forwarding pointer.
    addr = Builder.CreateStructGEP(addr, 1, "byref.forwarding");
    addr = Builder.CreateLoad(addr, "byref.addr.forwarded");

    // Cast back to byref* and GEP over to the actual object.
    addr = Builder.CreateBitCast(addr, byrefPointerType);
    addr = Builder.CreateStructGEP(addr, getByRefValueLLVMField(variable), 
                                   variable->getNameAsString());
  }

  if (variable->getType()->isReferenceType())
    addr = Builder.CreateLoad(addr, "ref.tmp");

  return addr;
}

llvm::Constant *
CodeGenModule::GetAddrOfGlobalBlock(const BlockExpr *blockExpr,
                                    const char *name) {
  CGBlockInfo blockInfo(blockExpr->getBlockDecl(), name);
  blockInfo.BlockExpression = blockExpr;

  // Compute information about the layout, etc., of this block.
  computeBlockInfo(*this, nullptr, blockInfo);

  // Using that metadata, generate the actual block function.
  llvm::Constant *blockFn;
  {
    llvm::DenseMap<const Decl*, llvm::Value*> LocalDeclMap;
    blockFn = CodeGenFunction(*this).GenerateBlockFunction(GlobalDecl(),
                                                           blockInfo,
                                                           LocalDeclMap,
                                                           false);
  }
  blockFn = llvm::ConstantExpr::getBitCast(blockFn, VoidPtrTy);

  return buildGlobalBlock(*this, blockInfo, blockFn);
}

static llvm::Constant *buildGlobalBlock(CodeGenModule &CGM,
                                        const CGBlockInfo &blockInfo,
                                        llvm::Constant *blockFn) {
  assert(blockInfo.CanBeGlobal);

  // Generate the constants for the block literal initializer.
  llvm::Constant *fields[BlockHeaderSize];

  // isa
  fields[0] = CGM.getNSConcreteGlobalBlock();

  // __flags
  BlockFlags flags = BLOCK_IS_GLOBAL | BLOCK_HAS_SIGNATURE;
  if (blockInfo.UsesStret) flags |= BLOCK_USE_STRET;
                                      
  fields[1] = llvm::ConstantInt::get(CGM.IntTy, flags.getBitMask());

  // Reserved
  fields[2] = llvm::Constant::getNullValue(CGM.IntTy);

  // Function
  fields[3] = blockFn;

  // Descriptor
  fields[4] = buildBlockDescriptor(CGM, blockInfo);

  llvm::Constant *init = llvm::ConstantStruct::getAnon(fields);

  llvm::GlobalVariable *literal =
    new llvm::GlobalVariable(CGM.getModule(),
                             init->getType(),
                             /*constant*/ true,
                             llvm::GlobalVariable::InternalLinkage,
                             init,
                             "__block_literal_global");
  literal->setAlignment(blockInfo.BlockAlign.getQuantity());

  // Return a constant of the appropriately-casted type.
  llvm::Type *requiredType =
    CGM.getTypes().ConvertType(blockInfo.getBlockExpr()->getType());
  return llvm::ConstantExpr::getBitCast(literal, requiredType);
}

llvm::Function *
CodeGenFunction::GenerateBlockFunction(GlobalDecl GD,
                                       const CGBlockInfo &blockInfo,
                                       const DeclMapTy &ldm,
                                       bool IsLambdaConversionToBlock) {
  const BlockDecl *blockDecl = blockInfo.getBlockDecl();

  CurGD = GD;
  
  BlockInfo = &blockInfo;

  // Arrange for local static and local extern declarations to appear
  // to be local to this function as well, in case they're directly
  // referenced in a block.
  for (DeclMapTy::const_iterator i = ldm.begin(), e = ldm.end(); i != e; ++i) {
    const auto *var = dyn_cast<VarDecl>(i->first);
    if (var && !var->hasLocalStorage())
      LocalDeclMap[var] = i->second;
  }

  // Begin building the function declaration.

  // Build the argument list.
  FunctionArgList args;

  // The first argument is the block pointer.  Just take it as a void*
  // and cast it later.
  QualType selfTy = getContext().VoidPtrTy;
  IdentifierInfo *II = &CGM.getContext().Idents.get(".block_descriptor");

  ImplicitParamDecl selfDecl(getContext(), const_cast<BlockDecl*>(blockDecl),
                             SourceLocation(), II, selfTy);
  args.push_back(&selfDecl);

  // Now add the rest of the parameters.
  for (auto i : blockDecl->params())
    args.push_back(i);

  // Create the function declaration.
  const FunctionProtoType *fnType = blockInfo.getBlockExpr()->getFunctionType();
  const CGFunctionInfo &fnInfo = CGM.getTypes().arrangeFreeFunctionDeclaration(
      fnType->getReturnType(), args, fnType->getExtInfo(),
      fnType->isVariadic());
  if (CGM.ReturnSlotInterferesWithArgs(fnInfo))
    blockInfo.UsesStret = true;

  llvm::FunctionType *fnLLVMType = CGM.getTypes().GetFunctionType(fnInfo);

  StringRef name = CGM.getBlockMangledName(GD, blockDecl);
  llvm::Function *fn = llvm::Function::Create(
      fnLLVMType, llvm::GlobalValue::InternalLinkage, name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(blockDecl, fn, fnInfo);

  // Begin generating the function.
  StartFunction(blockDecl, fnType->getReturnType(), fn, fnInfo, args,
                blockDecl->getLocation(),
                blockInfo.getBlockExpr()->getBody()->getLocStart());

  // Okay.  Undo some of what StartFunction did.
  
  // Pull the 'self' reference out of the local decl map.
  llvm::Value *blockAddr = LocalDeclMap[&selfDecl];
  LocalDeclMap.erase(&selfDecl);
  BlockPointer = Builder.CreateBitCast(blockAddr,
                                       blockInfo.StructureType->getPointerTo(),
                                       "block");
  // At -O0 we generate an explicit alloca for the BlockPointer, so the RA
  // won't delete the dbg.declare intrinsics for captured variables.
  llvm::Value *BlockPointerDbgLoc = BlockPointer;
  if (CGM.getCodeGenOpts().OptimizationLevel == 0) {
    // Allocate a stack slot for it, so we can point the debugger to it
    llvm::AllocaInst *Alloca = CreateTempAlloca(BlockPointer->getType(),
                                                "block.addr");
    unsigned Align = getContext().getDeclAlign(&selfDecl).getQuantity();
    Alloca->setAlignment(Align);
    // Set the DebugLocation to empty, so the store is recognized as a
    // frame setup instruction by llvm::DwarfDebug::beginFunction().
    NoLocation NL(*this, Builder);
    Builder.CreateAlignedStore(BlockPointer, Alloca, Align);
    BlockPointerDbgLoc = Alloca;
  }

  // If we have a C++ 'this' reference, go ahead and force it into
  // existence now.
  if (blockDecl->capturesCXXThis()) {
    llvm::Value *addr = Builder.CreateStructGEP(BlockPointer,
                                                blockInfo.CXXThisIndex,
                                                "block.captured-this");
    CXXThisValue = Builder.CreateLoad(addr, "this");
  }

  // Also force all the constant captures.
  for (const auto &CI : blockDecl->captures()) {
    const VarDecl *variable = CI.getVariable();
    const CGBlockInfo::Capture &capture = blockInfo.getCapture(variable);
    if (!capture.isConstant()) continue;

    unsigned align = getContext().getDeclAlign(variable).getQuantity();

    llvm::AllocaInst *alloca =
      CreateMemTemp(variable->getType(), "block.captured-const");
    alloca->setAlignment(align);

    Builder.CreateAlignedStore(capture.getConstant(), alloca, align);

    LocalDeclMap[variable] = alloca;
  }

  // Save a spot to insert the debug information for all the DeclRefExprs.
  llvm::BasicBlock *entry = Builder.GetInsertBlock();
  llvm::BasicBlock::iterator entry_ptr = Builder.GetInsertPoint();
  --entry_ptr;

  if (IsLambdaConversionToBlock)
    EmitLambdaBlockInvokeBody();
  else {
    PGO.assignRegionCounters(blockDecl, fn);
    RegionCounter Cnt = getPGORegionCounter(blockDecl->getBody());
    Cnt.beginRegion(Builder);
    EmitStmt(blockDecl->getBody());
    PGO.emitInstrumentationData();
    PGO.destroyRegionCounters();
  }

  // Remember where we were...
  llvm::BasicBlock *resume = Builder.GetInsertBlock();

  // Go back to the entry.
  ++entry_ptr;
  Builder.SetInsertPoint(entry, entry_ptr);

  // Emit debug information for all the DeclRefExprs.
  // FIXME: also for 'this'
  if (CGDebugInfo *DI = getDebugInfo()) {
    for (const auto &CI : blockDecl->captures()) {
      const VarDecl *variable = CI.getVariable();
      DI->EmitLocation(Builder, variable->getLocation());

      if (CGM.getCodeGenOpts().getDebugInfo()
            >= CodeGenOptions::LimitedDebugInfo) {
        const CGBlockInfo::Capture &capture = blockInfo.getCapture(variable);
        if (capture.isConstant()) {
          DI->EmitDeclareOfAutoVariable(variable, LocalDeclMap[variable],
                                        Builder);
          continue;
        }

        DI->EmitDeclareOfBlockDeclRefVariable(variable, BlockPointerDbgLoc,
                                              Builder, blockInfo);
      }
    }
    // Recover location if it was changed in the above loop.
    DI->EmitLocation(Builder,
                     cast<CompoundStmt>(blockDecl->getBody())->getRBracLoc());
  }

  // And resume where we left off.
  if (resume == nullptr)
    Builder.ClearInsertionPoint();
  else
    Builder.SetInsertPoint(resume);

  FinishFunction(cast<CompoundStmt>(blockDecl->getBody())->getRBracLoc());

  return fn;
}

/*
    notes.push_back(HelperInfo());
    HelperInfo &note = notes.back();
    note.index = capture.getIndex();
    note.RequiresCopying = (ci->hasCopyExpr() || BlockRequiresCopying(type));
    note.cxxbar_import = ci->getCopyExpr();

    if (ci->isByRef()) {
      note.flag = BLOCK_FIELD_IS_BYREF;
      if (type.isObjCGCWeak())
        note.flag |= BLOCK_FIELD_IS_WEAK;
    } else if (type->isBlockPointerType()) {
      note.flag = BLOCK_FIELD_IS_BLOCK;
    } else {
      note.flag = BLOCK_FIELD_IS_OBJECT;
    }
 */


/// Generate the copy-helper function for a block closure object:
///   static void block_copy_helper(block_t *dst, block_t *src);
/// The runtime will have previously initialized 'dst' by doing a
/// bit-copy of 'src'.
///
/// Note that this copies an entire block closure object to the heap;
/// it should not be confused with a 'byref copy helper', which moves
/// the contents of an individual __block variable to the heap.
llvm::Constant *
CodeGenFunction::GenerateCopyHelperFunction(const CGBlockInfo &blockInfo) {
  ASTContext &C = getContext();

  FunctionArgList args;
  ImplicitParamDecl dstDecl(getContext(), nullptr, SourceLocation(), nullptr,
                            C.VoidPtrTy);
  args.push_back(&dstDecl);
  ImplicitParamDecl srcDecl(getContext(), nullptr, SourceLocation(), nullptr,
                            C.VoidPtrTy);
  args.push_back(&srcDecl);

  const CGFunctionInfo &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
      C.VoidTy, args, FunctionType::ExtInfo(), /*variadic=*/false);

  // FIXME: it would be nice if these were mergeable with things with
  // identical semantics.
  llvm::FunctionType *LTy = CGM.getTypes().GetFunctionType(FI);

  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           "__copy_helper_block_", &CGM.getModule());

  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__copy_helper_block_");

  FunctionDecl *FD = FunctionDecl::Create(C,
                                          C.getTranslationUnitDecl(),
                                          SourceLocation(),
                                          SourceLocation(), II, C.VoidTy,
                                          nullptr, SC_Static,
                                          false,
                                          false);
  // Create a scope with an artificial location for the body of this function.
  ArtificialLocation AL(*this, Builder);
  StartFunction(FD, C.VoidTy, Fn, FI, args);
  AL.Emit();

  llvm::Type *structPtrTy = blockInfo.StructureType->getPointerTo();

  llvm::Value *src = GetAddrOfLocalVar(&srcDecl);
  src = Builder.CreateLoad(src);
  src = Builder.CreateBitCast(src, structPtrTy, "block.source");

  llvm::Value *dst = GetAddrOfLocalVar(&dstDecl);
  dst = Builder.CreateLoad(dst);
  dst = Builder.CreateBitCast(dst, structPtrTy, "block.dest");

  const BlockDecl *blockDecl = blockInfo.getBlockDecl();

  for (const auto &CI : blockDecl->captures()) {
    const VarDecl *variable = CI.getVariable();
    QualType type = variable->getType();

    const CGBlockInfo::Capture &capture = blockInfo.getCapture(variable);
    if (capture.isConstant()) continue;

    const Expr *copyExpr = CI.getCopyExpr();
    BlockFieldFlags flags;

    bool useARCWeakCopy = false;
    bool useARCStrongCopy = false;

    if (copyExpr) {
      assert(!CI.isByRef());
      // don't bother computing flags

    } else if (CI.isByRef()) {
      flags = BLOCK_FIELD_IS_BYREF;
      if (type.isObjCGCWeak())
        flags |= BLOCK_FIELD_IS_WEAK;

    } else if (type->isObjCRetainableType()) {
      flags = BLOCK_FIELD_IS_OBJECT;
      bool isBlockPointer = type->isBlockPointerType();
      if (isBlockPointer)
        flags = BLOCK_FIELD_IS_BLOCK;

      // Special rules for ARC captures:
      if (getLangOpts().ObjCAutoRefCount) {
        Qualifiers qs = type.getQualifiers();

        // We need to register __weak direct captures with the runtime.
        if (qs.getObjCLifetime() == Qualifiers::OCL_Weak) {
          useARCWeakCopy = true;

        // We need to retain the copied value for __strong direct captures.
        } else if (qs.getObjCLifetime() == Qualifiers::OCL_Strong) {
          // If it's a block pointer, we have to copy the block and
          // assign that to the destination pointer, so we might as
          // well use _Block_object_assign.  Otherwise we can avoid that.
          if (!isBlockPointer)
            useARCStrongCopy = true;

        // Otherwise the memcpy is fine.
        } else {
          continue;
        }

      // Non-ARC captures of retainable pointers are strong and
      // therefore require a call to _Block_object_assign.
      } else {
        // fall through
      }
    } else {
      continue;
    }

    unsigned index = capture.getIndex();
    llvm::Value *srcField = Builder.CreateStructGEP(src, index);
    llvm::Value *dstField = Builder.CreateStructGEP(dst, index);

    // If there's an explicit copy expression, we do that.
    if (copyExpr) {
      EmitSynthesizedCXXCopyCtor(dstField, srcField, copyExpr);
    } else if (useARCWeakCopy) {
      EmitARCCopyWeak(dstField, srcField);
    } else {
      llvm::Value *srcValue = Builder.CreateLoad(srcField, "blockcopy.src");
      if (useARCStrongCopy) {
        // At -O0, store null into the destination field (so that the
        // storeStrong doesn't over-release) and then call storeStrong.
        // This is a workaround to not having an initStrong call.
        if (CGM.getCodeGenOpts().OptimizationLevel == 0) {
          auto *ty = cast<llvm::PointerType>(srcValue->getType());
          llvm::Value *null = llvm::ConstantPointerNull::get(ty);
          Builder.CreateStore(null, dstField);
          EmitARCStoreStrongCall(dstField, srcValue, true);

        // With optimization enabled, take advantage of the fact that
        // the blocks runtime guarantees a memcpy of the block data, and
        // just emit a retain of the src field.
        } else {
          EmitARCRetainNonBlock(srcValue);

          // We don't need this anymore, so kill it.  It's not quite
          // worth the annoyance to avoid creating it in the first place.
          cast<llvm::Instruction>(dstField)->eraseFromParent();
        }
      } else {
        srcValue = Builder.CreateBitCast(srcValue, VoidPtrTy);
        llvm::Value *dstAddr = Builder.CreateBitCast(dstField, VoidPtrTy);
        llvm::Value *args[] = {
          dstAddr, srcValue, llvm::ConstantInt::get(Int32Ty, flags.getBitMask())
        };

        bool copyCanThrow = false;
        if (CI.isByRef() && variable->getType()->getAsCXXRecordDecl()) {
          const Expr *copyExpr =
            CGM.getContext().getBlockVarCopyInits(variable);
          if (copyExpr) {
            copyCanThrow = true; // FIXME: reuse the noexcept logic
          }
        }

        if (copyCanThrow) {
          EmitRuntimeCallOrInvoke(CGM.getBlockObjectAssign(), args);
        } else {
          EmitNounwindRuntimeCall(CGM.getBlockObjectAssign(), args);
        }
      }
    }
  }

  FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, VoidPtrTy);
}

/// Generate the destroy-helper function for a block closure object:
///   static void block_destroy_helper(block_t *theBlock);
///
/// Note that this destroys a heap-allocated block closure object;
/// it should not be confused with a 'byref destroy helper', which
/// destroys the heap-allocated contents of an individual __block
/// variable.
llvm::Constant *
CodeGenFunction::GenerateDestroyHelperFunction(const CGBlockInfo &blockInfo) {
  ASTContext &C = getContext();

  FunctionArgList args;
  ImplicitParamDecl srcDecl(getContext(), nullptr, SourceLocation(), nullptr,
                            C.VoidPtrTy);
  args.push_back(&srcDecl);

  const CGFunctionInfo &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
      C.VoidTy, args, FunctionType::ExtInfo(), /*variadic=*/false);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  llvm::FunctionType *LTy = CGM.getTypes().GetFunctionType(FI);

  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           "__destroy_helper_block_", &CGM.getModule());

  IdentifierInfo *II
    = &CGM.getContext().Idents.get("__destroy_helper_block_");

  FunctionDecl *FD = FunctionDecl::Create(C, C.getTranslationUnitDecl(),
                                          SourceLocation(),
                                          SourceLocation(), II, C.VoidTy,
                                          nullptr, SC_Static,
                                          false, false);
  // Create a scope with an artificial location for the body of this function.
  ArtificialLocation AL(*this, Builder);
  StartFunction(FD, C.VoidTy, Fn, FI, args);
  AL.Emit();

  llvm::Type *structPtrTy = blockInfo.StructureType->getPointerTo();

  llvm::Value *src = GetAddrOfLocalVar(&srcDecl);
  src = Builder.CreateLoad(src);
  src = Builder.CreateBitCast(src, structPtrTy, "block");

  const BlockDecl *blockDecl = blockInfo.getBlockDecl();

  CodeGenFunction::RunCleanupsScope cleanups(*this);

  for (const auto &CI : blockDecl->captures()) {
    const VarDecl *variable = CI.getVariable();
    QualType type = variable->getType();

    const CGBlockInfo::Capture &capture = blockInfo.getCapture(variable);
    if (capture.isConstant()) continue;

    BlockFieldFlags flags;
    const CXXDestructorDecl *dtor = nullptr;

    bool useARCWeakDestroy = false;
    bool useARCStrongDestroy = false;

    if (CI.isByRef()) {
      flags = BLOCK_FIELD_IS_BYREF;
      if (type.isObjCGCWeak())
        flags |= BLOCK_FIELD_IS_WEAK;
    } else if (const CXXRecordDecl *record = type->getAsCXXRecordDecl()) {
      if (record->hasTrivialDestructor())
        continue;
      dtor = record->getDestructor();
    } else if (type->isObjCRetainableType()) {
      flags = BLOCK_FIELD_IS_OBJECT;
      if (type->isBlockPointerType())
        flags = BLOCK_FIELD_IS_BLOCK;

      // Special rules for ARC captures.
      if (getLangOpts().ObjCAutoRefCount) {
        Qualifiers qs = type.getQualifiers();

        // Don't generate special dispose logic for a captured object
        // unless it's __strong or __weak.
        if (!qs.hasStrongOrWeakObjCLifetime())
          continue;

        // Support __weak direct captures.
        if (qs.getObjCLifetime() == Qualifiers::OCL_Weak)
          useARCWeakDestroy = true;

        // Tools really want us to use objc_storeStrong here.
        else
          useARCStrongDestroy = true;
      }
    } else {
      continue;
    }

    unsigned index = capture.getIndex();
    llvm::Value *srcField = Builder.CreateStructGEP(src, index);

    // If there's an explicit copy expression, we do that.
    if (dtor) {
      PushDestructorCleanup(dtor, srcField);

    // If this is a __weak capture, emit the release directly.
    } else if (useARCWeakDestroy) {
      EmitARCDestroyWeak(srcField);

    // Destroy strong objects with a call if requested.
    } else if (useARCStrongDestroy) {
      EmitARCDestroyStrong(srcField, ARCImpreciseLifetime);

    // Otherwise we call _Block_object_dispose.  It wouldn't be too
    // hard to just emit this as a cleanup if we wanted to make sure
    // that things were done in reverse.
    } else {
      llvm::Value *value = Builder.CreateLoad(srcField);
      value = Builder.CreateBitCast(value, VoidPtrTy);
      BuildBlockRelease(value, flags);
    }
  }

  cleanups.ForceCleanup();

  FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, VoidPtrTy);
}

namespace {

/// Emits the copy/dispose helper functions for a __block object of id type.
class ObjectByrefHelpers : public CodeGenModule::ByrefHelpers {
  BlockFieldFlags Flags;

public:
  ObjectByrefHelpers(CharUnits alignment, BlockFieldFlags flags)
    : ByrefHelpers(alignment), Flags(flags) {}

  void emitCopy(CodeGenFunction &CGF, llvm::Value *destField,
                llvm::Value *srcField) override {
    destField = CGF.Builder.CreateBitCast(destField, CGF.VoidPtrTy);

    srcField = CGF.Builder.CreateBitCast(srcField, CGF.VoidPtrPtrTy);
    llvm::Value *srcValue = CGF.Builder.CreateLoad(srcField);

    unsigned flags = (Flags | BLOCK_BYREF_CALLER).getBitMask();

    llvm::Value *flagsVal = llvm::ConstantInt::get(CGF.Int32Ty, flags);
    llvm::Value *fn = CGF.CGM.getBlockObjectAssign();

    llvm::Value *args[] = { destField, srcValue, flagsVal };
    CGF.EmitNounwindRuntimeCall(fn, args);
  }

  void emitDispose(CodeGenFunction &CGF, llvm::Value *field) override {
    field = CGF.Builder.CreateBitCast(field, CGF.Int8PtrTy->getPointerTo(0));
    llvm::Value *value = CGF.Builder.CreateLoad(field);

    CGF.BuildBlockRelease(value, Flags | BLOCK_BYREF_CALLER);
  }

  void profileImpl(llvm::FoldingSetNodeID &id) const override {
    id.AddInteger(Flags.getBitMask());
  }
};

/// Emits the copy/dispose helpers for an ARC __block __weak variable.
class ARCWeakByrefHelpers : public CodeGenModule::ByrefHelpers {
public:
  ARCWeakByrefHelpers(CharUnits alignment) : ByrefHelpers(alignment) {}

  void emitCopy(CodeGenFunction &CGF, llvm::Value *destField,
                llvm::Value *srcField) override {
    CGF.EmitARCMoveWeak(destField, srcField);
  }

  void emitDispose(CodeGenFunction &CGF, llvm::Value *field) override {
    CGF.EmitARCDestroyWeak(field);
  }

  void profileImpl(llvm::FoldingSetNodeID &id) const override {
    // 0 is distinguishable from all pointers and byref flags
    id.AddInteger(0);
  }
};

/// Emits the copy/dispose helpers for an ARC __block __strong variable
/// that's not of block-pointer type.
class ARCStrongByrefHelpers : public CodeGenModule::ByrefHelpers {
public:
  ARCStrongByrefHelpers(CharUnits alignment) : ByrefHelpers(alignment) {}

  void emitCopy(CodeGenFunction &CGF, llvm::Value *destField,
                llvm::Value *srcField) override {
    // Do a "move" by copying the value and then zeroing out the old
    // variable.

    llvm::LoadInst *value = CGF.Builder.CreateLoad(srcField);
    value->setAlignment(Alignment.getQuantity());
    
    llvm::Value *null =
      llvm::ConstantPointerNull::get(cast<llvm::PointerType>(value->getType()));

    if (CGF.CGM.getCodeGenOpts().OptimizationLevel == 0) {
      llvm::StoreInst *store = CGF.Builder.CreateStore(null, destField);
      store->setAlignment(Alignment.getQuantity());
      CGF.EmitARCStoreStrongCall(destField, value, /*ignored*/ true);
      CGF.EmitARCStoreStrongCall(srcField, null, /*ignored*/ true);
      return;
    }
    llvm::StoreInst *store = CGF.Builder.CreateStore(value, destField);
    store->setAlignment(Alignment.getQuantity());

    store = CGF.Builder.CreateStore(null, srcField);
    store->setAlignment(Alignment.getQuantity());
  }

  void emitDispose(CodeGenFunction &CGF, llvm::Value *field) override {
    CGF.EmitARCDestroyStrong(field, ARCImpreciseLifetime);
  }

  void profileImpl(llvm::FoldingSetNodeID &id) const override {
    // 1 is distinguishable from all pointers and byref flags
    id.AddInteger(1);
  }
};

/// Emits the copy/dispose helpers for an ARC __block __strong
/// variable that's of block-pointer type.
class ARCStrongBlockByrefHelpers : public CodeGenModule::ByrefHelpers {
public:
  ARCStrongBlockByrefHelpers(CharUnits alignment) : ByrefHelpers(alignment) {}

  void emitCopy(CodeGenFunction &CGF, llvm::Value *destField,
                llvm::Value *srcField) override {
    // Do the copy with objc_retainBlock; that's all that
    // _Block_object_assign would do anyway, and we'd have to pass the
    // right arguments to make sure it doesn't get no-op'ed.
    llvm::LoadInst *oldValue = CGF.Builder.CreateLoad(srcField);
    oldValue->setAlignment(Alignment.getQuantity());

    llvm::Value *copy = CGF.EmitARCRetainBlock(oldValue, /*mandatory*/ true);

    llvm::StoreInst *store = CGF.Builder.CreateStore(copy, destField);
    store->setAlignment(Alignment.getQuantity());
  }

  void emitDispose(CodeGenFunction &CGF, llvm::Value *field) override {
    CGF.EmitARCDestroyStrong(field, ARCImpreciseLifetime);
  }

  void profileImpl(llvm::FoldingSetNodeID &id) const override {
    // 2 is distinguishable from all pointers and byref flags
    id.AddInteger(2);
  }
};

/// Emits the copy/dispose helpers for a __block variable with a
/// nontrivial copy constructor or destructor.
class CXXByrefHelpers : public CodeGenModule::ByrefHelpers {
  QualType VarType;
  const Expr *CopyExpr;

public:
  CXXByrefHelpers(CharUnits alignment, QualType type,
                  const Expr *copyExpr)
    : ByrefHelpers(alignment), VarType(type), CopyExpr(copyExpr) {}

  bool needsCopy() const override { return CopyExpr != nullptr; }
  void emitCopy(CodeGenFunction &CGF, llvm::Value *destField,
                llvm::Value *srcField) override {
    if (!CopyExpr) return;
    CGF.EmitSynthesizedCXXCopyCtor(destField, srcField, CopyExpr);
  }

  void emitDispose(CodeGenFunction &CGF, llvm::Value *field) override {
    EHScopeStack::stable_iterator cleanupDepth = CGF.EHStack.stable_begin();
    CGF.PushDestructorCleanup(VarType, field);
    CGF.PopCleanupBlocks(cleanupDepth);
  }

  void profileImpl(llvm::FoldingSetNodeID &id) const override {
    id.AddPointer(VarType.getCanonicalType().getAsOpaquePtr());
  }
};
} // end anonymous namespace

static llvm::Constant *
generateByrefCopyHelper(CodeGenFunction &CGF,
                        llvm::StructType &byrefType,
                        unsigned valueFieldIndex,
                        CodeGenModule::ByrefHelpers &byrefInfo) {
  ASTContext &Context = CGF.getContext();

  QualType R = Context.VoidTy;

  FunctionArgList args;
  ImplicitParamDecl dst(CGF.getContext(), nullptr, SourceLocation(), nullptr,
                        Context.VoidPtrTy);
  args.push_back(&dst);

  ImplicitParamDecl src(CGF.getContext(), nullptr, SourceLocation(), nullptr,
                        Context.VoidPtrTy);
  args.push_back(&src);

  const CGFunctionInfo &FI = CGF.CGM.getTypes().arrangeFreeFunctionDeclaration(
      R, args, FunctionType::ExtInfo(), /*variadic=*/false);

  CodeGenTypes &Types = CGF.CGM.getTypes();
  llvm::FunctionType *LTy = Types.GetFunctionType(FI);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           "__Block_byref_object_copy_", &CGF.CGM.getModule());

  IdentifierInfo *II
    = &Context.Idents.get("__Block_byref_object_copy_");

  FunctionDecl *FD = FunctionDecl::Create(Context,
                                          Context.getTranslationUnitDecl(),
                                          SourceLocation(),
                                          SourceLocation(), II, R, nullptr,
                                          SC_Static,
                                          false, false);

  CGF.StartFunction(FD, R, Fn, FI, args);

  if (byrefInfo.needsCopy()) {
    llvm::Type *byrefPtrType = byrefType.getPointerTo(0);

    // dst->x
    llvm::Value *destField = CGF.GetAddrOfLocalVar(&dst);
    destField = CGF.Builder.CreateLoad(destField);
    destField = CGF.Builder.CreateBitCast(destField, byrefPtrType);
    destField = CGF.Builder.CreateStructGEP(destField, valueFieldIndex, "x");

    // src->x
    llvm::Value *srcField = CGF.GetAddrOfLocalVar(&src);
    srcField = CGF.Builder.CreateLoad(srcField);
    srcField = CGF.Builder.CreateBitCast(srcField, byrefPtrType);
    srcField = CGF.Builder.CreateStructGEP(srcField, valueFieldIndex, "x");

    byrefInfo.emitCopy(CGF, destField, srcField);
  }  

  CGF.FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, CGF.Int8PtrTy);
}

/// Build the copy helper for a __block variable.
static llvm::Constant *buildByrefCopyHelper(CodeGenModule &CGM,
                                            llvm::StructType &byrefType,
                                            unsigned byrefValueIndex,
                                            CodeGenModule::ByrefHelpers &info) {
  CodeGenFunction CGF(CGM);
  return generateByrefCopyHelper(CGF, byrefType, byrefValueIndex, info);
}

/// Generate code for a __block variable's dispose helper.
static llvm::Constant *
generateByrefDisposeHelper(CodeGenFunction &CGF,
                           llvm::StructType &byrefType,
                           unsigned byrefValueIndex,
                           CodeGenModule::ByrefHelpers &byrefInfo) {
  ASTContext &Context = CGF.getContext();
  QualType R = Context.VoidTy;

  FunctionArgList args;
  ImplicitParamDecl src(CGF.getContext(), nullptr, SourceLocation(), nullptr,
                        Context.VoidPtrTy);
  args.push_back(&src);

  const CGFunctionInfo &FI = CGF.CGM.getTypes().arrangeFreeFunctionDeclaration(
      R, args, FunctionType::ExtInfo(), /*variadic=*/false);

  CodeGenTypes &Types = CGF.CGM.getTypes();
  llvm::FunctionType *LTy = Types.GetFunctionType(FI);

  // FIXME: We'd like to put these into a mergable by content, with
  // internal linkage.
  llvm::Function *Fn =
    llvm::Function::Create(LTy, llvm::GlobalValue::InternalLinkage,
                           "__Block_byref_object_dispose_",
                           &CGF.CGM.getModule());

  IdentifierInfo *II
    = &Context.Idents.get("__Block_byref_object_dispose_");

  FunctionDecl *FD = FunctionDecl::Create(Context,
                                          Context.getTranslationUnitDecl(),
                                          SourceLocation(),
                                          SourceLocation(), II, R, nullptr,
                                          SC_Static,
                                          false, false);
  CGF.StartFunction(FD, R, Fn, FI, args);

  if (byrefInfo.needsDispose()) {
    llvm::Value *V = CGF.GetAddrOfLocalVar(&src);
    V = CGF.Builder.CreateLoad(V);
    V = CGF.Builder.CreateBitCast(V, byrefType.getPointerTo(0));
    V = CGF.Builder.CreateStructGEP(V, byrefValueIndex, "x");

    byrefInfo.emitDispose(CGF, V);
  }

  CGF.FinishFunction();

  return llvm::ConstantExpr::getBitCast(Fn, CGF.Int8PtrTy);
}

/// Build the dispose helper for a __block variable.
static llvm::Constant *buildByrefDisposeHelper(CodeGenModule &CGM,
                                              llvm::StructType &byrefType,
                                               unsigned byrefValueIndex,
                                            CodeGenModule::ByrefHelpers &info) {
  CodeGenFunction CGF(CGM);
  return generateByrefDisposeHelper(CGF, byrefType, byrefValueIndex, info);
}

/// Lazily build the copy and dispose helpers for a __block variable
/// with the given information.
template <class T> static T *buildByrefHelpers(CodeGenModule &CGM,
                                               llvm::StructType &byrefTy,
                                               unsigned byrefValueIndex,
                                               T &byrefInfo) {
  // Increase the field's alignment to be at least pointer alignment,
  // since the layout of the byref struct will guarantee at least that.
  byrefInfo.Alignment = std::max(byrefInfo.Alignment,
                              CharUnits::fromQuantity(CGM.PointerAlignInBytes));

  llvm::FoldingSetNodeID id;
  byrefInfo.Profile(id);

  void *insertPos;
  CodeGenModule::ByrefHelpers *node
    = CGM.ByrefHelpersCache.FindNodeOrInsertPos(id, insertPos);
  if (node) return static_cast<T*>(node);

  byrefInfo.CopyHelper =
    buildByrefCopyHelper(CGM, byrefTy, byrefValueIndex, byrefInfo);
  byrefInfo.DisposeHelper =
    buildByrefDisposeHelper(CGM, byrefTy, byrefValueIndex,byrefInfo);

  T *copy = new (CGM.getContext()) T(byrefInfo);
  CGM.ByrefHelpersCache.InsertNode(copy, insertPos);
  return copy;
}

/// Build the copy and dispose helpers for the given __block variable
/// emission.  Places the helpers in the global cache.  Returns null
/// if no helpers are required.
CodeGenModule::ByrefHelpers *
CodeGenFunction::buildByrefHelpers(llvm::StructType &byrefType,
                                   const AutoVarEmission &emission) {
  const VarDecl &var = *emission.Variable;
  QualType type = var.getType();

  unsigned byrefValueIndex = getByRefValueLLVMField(&var);

  if (const CXXRecordDecl *record = type->getAsCXXRecordDecl()) {
    const Expr *copyExpr = CGM.getContext().getBlockVarCopyInits(&var);
    if (!copyExpr && record->hasTrivialDestructor()) return nullptr;

    CXXByrefHelpers byrefInfo(emission.Alignment, type, copyExpr);
    return ::buildByrefHelpers(CGM, byrefType, byrefValueIndex, byrefInfo);
  }

  // Otherwise, if we don't have a retainable type, there's nothing to do.
  // that the runtime does extra copies.
  if (!type->isObjCRetainableType()) return nullptr;

  Qualifiers qs = type.getQualifiers();

  // If we have lifetime, that dominates.
  if (Qualifiers::ObjCLifetime lifetime = qs.getObjCLifetime()) {
    assert(getLangOpts().ObjCAutoRefCount);

    switch (lifetime) {
    case Qualifiers::OCL_None: llvm_unreachable("impossible");

    // These are just bits as far as the runtime is concerned.
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Autoreleasing:
      return nullptr;

    // Tell the runtime that this is ARC __weak, called by the
    // byref routines.
    case Qualifiers::OCL_Weak: {
      ARCWeakByrefHelpers byrefInfo(emission.Alignment);
      return ::buildByrefHelpers(CGM, byrefType, byrefValueIndex, byrefInfo);
    }

    // ARC __strong __block variables need to be retained.
    case Qualifiers::OCL_Strong:
      // Block pointers need to be copied, and there's no direct
      // transfer possible.
      if (type->isBlockPointerType()) {
        ARCStrongBlockByrefHelpers byrefInfo(emission.Alignment);
        return ::buildByrefHelpers(CGM, byrefType, byrefValueIndex, byrefInfo);

      // Otherwise, we transfer ownership of the retain from the stack
      // to the heap.
      } else {
        ARCStrongByrefHelpers byrefInfo(emission.Alignment);
        return ::buildByrefHelpers(CGM, byrefType, byrefValueIndex, byrefInfo);
      }
    }
    llvm_unreachable("fell out of lifetime switch!");
  }

  BlockFieldFlags flags;
  if (type->isBlockPointerType()) {
    flags |= BLOCK_FIELD_IS_BLOCK;
  } else if (CGM.getContext().isObjCNSObjectType(type) || 
             type->isObjCObjectPointerType()) {
    flags |= BLOCK_FIELD_IS_OBJECT;
  } else {
    return nullptr;
  }

  if (type.isObjCGCWeak())
    flags |= BLOCK_FIELD_IS_WEAK;

  ObjectByrefHelpers byrefInfo(emission.Alignment, flags);
  return ::buildByrefHelpers(CGM, byrefType, byrefValueIndex, byrefInfo);
}

unsigned CodeGenFunction::getByRefValueLLVMField(const ValueDecl *VD) const {
  assert(ByRefValueInfo.count(VD) && "Did not find value!");
  
  return ByRefValueInfo.find(VD)->second.second;
}

llvm::Value *CodeGenFunction::BuildBlockByrefAddress(llvm::Value *BaseAddr,
                                                     const VarDecl *V) {
  llvm::Value *Loc = Builder.CreateStructGEP(BaseAddr, 1, "forwarding");
  Loc = Builder.CreateLoad(Loc);
  Loc = Builder.CreateStructGEP(Loc, getByRefValueLLVMField(V),
                                V->getNameAsString());
  return Loc;
}

/// BuildByRefType - This routine changes a __block variable declared as T x
///   into:
///
///      struct {
///        void *__isa;
///        void *__forwarding;
///        int32_t __flags;
///        int32_t __size;
///        void *__copy_helper;       // only if needed
///        void *__destroy_helper;    // only if needed
///        void *__byref_variable_layout;// only if needed
///        char padding[X];           // only if needed
///        T x;
///      } x
///
llvm::Type *CodeGenFunction::BuildByRefType(const VarDecl *D) {
  std::pair<llvm::Type *, unsigned> &Info = ByRefValueInfo[D];
  if (Info.first)
    return Info.first;
  
  QualType Ty = D->getType();

  SmallVector<llvm::Type *, 8> types;
  
  llvm::StructType *ByRefType =
    llvm::StructType::create(getLLVMContext(),
                             "struct.__block_byref_" + D->getNameAsString());
  
  // void *__isa;
  types.push_back(Int8PtrTy);
  
  // void *__forwarding;
  types.push_back(llvm::PointerType::getUnqual(ByRefType));
  
  // int32_t __flags;
  types.push_back(Int32Ty);
    
  // int32_t __size;
  types.push_back(Int32Ty);
  // Note that this must match *exactly* the logic in buildByrefHelpers.
  bool HasCopyAndDispose = getContext().BlockRequiresCopying(Ty, D);
  if (HasCopyAndDispose) {
    /// void *__copy_helper;
    types.push_back(Int8PtrTy);
    
    /// void *__destroy_helper;
    types.push_back(Int8PtrTy);
  }
  bool HasByrefExtendedLayout = false;
  Qualifiers::ObjCLifetime Lifetime;
  if (getContext().getByrefLifetime(Ty, Lifetime, HasByrefExtendedLayout) &&
      HasByrefExtendedLayout)
    /// void *__byref_variable_layout;
    types.push_back(Int8PtrTy);

  bool Packed = false;
  CharUnits Align = getContext().getDeclAlign(D);
  if (Align >
      getContext().toCharUnitsFromBits(getTarget().getPointerAlign(0))) {
    // We have to insert padding.
    
    // The struct above has 2 32-bit integers.
    unsigned CurrentOffsetInBytes = 4 * 2;
    
    // And either 2, 3, 4 or 5 pointers.
    unsigned noPointers = 2;
    if (HasCopyAndDispose)
      noPointers += 2;
    if (HasByrefExtendedLayout)
      noPointers += 1;
    
    CurrentOffsetInBytes += noPointers * CGM.getDataLayout().getTypeAllocSize(Int8PtrTy);
    
    // Align the offset.
    unsigned AlignedOffsetInBytes = 
      llvm::RoundUpToAlignment(CurrentOffsetInBytes, Align.getQuantity());
    
    unsigned NumPaddingBytes = AlignedOffsetInBytes - CurrentOffsetInBytes;
    if (NumPaddingBytes > 0) {
      llvm::Type *Ty = Int8Ty;
      // FIXME: We need a sema error for alignment larger than the minimum of
      // the maximal stack alignment and the alignment of malloc on the system.
      if (NumPaddingBytes > 1)
        Ty = llvm::ArrayType::get(Ty, NumPaddingBytes);
    
      types.push_back(Ty);

      // We want a packed struct.
      Packed = true;
    }
  }

  // T x;
  types.push_back(ConvertTypeForMem(Ty));
  
  ByRefType->setBody(types, Packed);
  
  Info.first = ByRefType;
  
  Info.second = types.size() - 1;
  
  return Info.first;
}

/// Initialize the structural components of a __block variable, i.e.
/// everything but the actual object.
void CodeGenFunction::emitByrefStructureInit(const AutoVarEmission &emission) {
  // Find the address of the local.
  llvm::Value *addr = emission.Address;

  // That's an alloca of the byref structure type.
  llvm::StructType *byrefType = cast<llvm::StructType>(
                 cast<llvm::PointerType>(addr->getType())->getElementType());

  // Build the byref helpers if necessary.  This is null if we don't need any.
  CodeGenModule::ByrefHelpers *helpers =
    buildByrefHelpers(*byrefType, emission);

  const VarDecl &D = *emission.Variable;
  QualType type = D.getType();

  bool HasByrefExtendedLayout;
  Qualifiers::ObjCLifetime ByrefLifetime;
  bool ByRefHasLifetime =
    getContext().getByrefLifetime(type, ByrefLifetime, HasByrefExtendedLayout);
  
  llvm::Value *V;

  // Initialize the 'isa', which is just 0 or 1.
  int isa = 0;
  if (type.isObjCGCWeak())
    isa = 1;
  V = Builder.CreateIntToPtr(Builder.getInt32(isa), Int8PtrTy, "isa");
  Builder.CreateStore(V, Builder.CreateStructGEP(addr, 0, "byref.isa"));

  // Store the address of the variable into its own forwarding pointer.
  Builder.CreateStore(addr,
                      Builder.CreateStructGEP(addr, 1, "byref.forwarding"));

  // Blocks ABI:
  //   c) the flags field is set to either 0 if no helper functions are
  //      needed or BLOCK_BYREF_HAS_COPY_DISPOSE if they are,
  BlockFlags flags;
  if (helpers) flags |= BLOCK_BYREF_HAS_COPY_DISPOSE;
  if (ByRefHasLifetime) {
    if (HasByrefExtendedLayout) flags |= BLOCK_BYREF_LAYOUT_EXTENDED;
      else switch (ByrefLifetime) {
        case Qualifiers::OCL_Strong:
          flags |= BLOCK_BYREF_LAYOUT_STRONG;
          break;
        case Qualifiers::OCL_Weak:
          flags |= BLOCK_BYREF_LAYOUT_WEAK;
          break;
        case Qualifiers::OCL_ExplicitNone:
          flags |= BLOCK_BYREF_LAYOUT_UNRETAINED;
          break;
        case Qualifiers::OCL_None:
          if (!type->isObjCObjectPointerType() && !type->isBlockPointerType())
            flags |= BLOCK_BYREF_LAYOUT_NON_OBJECT;
          break;
        default:
          break;
      }
    if (CGM.getLangOpts().ObjCGCBitmapPrint) {
      printf("\n Inline flag for BYREF variable layout (%d):", flags.getBitMask());
      if (flags & BLOCK_BYREF_HAS_COPY_DISPOSE)
        printf(" BLOCK_BYREF_HAS_COPY_DISPOSE");
      if (flags & BLOCK_BYREF_LAYOUT_MASK) {
        BlockFlags ThisFlag(flags.getBitMask() & BLOCK_BYREF_LAYOUT_MASK);
        if (ThisFlag ==  BLOCK_BYREF_LAYOUT_EXTENDED)
          printf(" BLOCK_BYREF_LAYOUT_EXTENDED");
        if (ThisFlag ==  BLOCK_BYREF_LAYOUT_STRONG)
          printf(" BLOCK_BYREF_LAYOUT_STRONG");
        if (ThisFlag == BLOCK_BYREF_LAYOUT_WEAK)
          printf(" BLOCK_BYREF_LAYOUT_WEAK");
        if (ThisFlag == BLOCK_BYREF_LAYOUT_UNRETAINED)
          printf(" BLOCK_BYREF_LAYOUT_UNRETAINED");
        if (ThisFlag == BLOCK_BYREF_LAYOUT_NON_OBJECT)
          printf(" BLOCK_BYREF_LAYOUT_NON_OBJECT");
      }
      printf("\n");
    }
  }
  
  Builder.CreateStore(llvm::ConstantInt::get(IntTy, flags.getBitMask()),
                      Builder.CreateStructGEP(addr, 2, "byref.flags"));

  CharUnits byrefSize = CGM.GetTargetTypeStoreSize(byrefType);
  V = llvm::ConstantInt::get(IntTy, byrefSize.getQuantity());
  Builder.CreateStore(V, Builder.CreateStructGEP(addr, 3, "byref.size"));

  if (helpers) {
    llvm::Value *copy_helper = Builder.CreateStructGEP(addr, 4);
    Builder.CreateStore(helpers->CopyHelper, copy_helper);

    llvm::Value *destroy_helper = Builder.CreateStructGEP(addr, 5);
    Builder.CreateStore(helpers->DisposeHelper, destroy_helper);
  }
  if (ByRefHasLifetime && HasByrefExtendedLayout) {
    llvm::Constant* ByrefLayoutInfo = CGM.getObjCRuntime().BuildByrefLayout(CGM, type);
    llvm::Value *ByrefInfoAddr = Builder.CreateStructGEP(addr, helpers ? 6 : 4,
                                                         "byref.layout");
    // cast destination to pointer to source type.
    llvm::Type *DesTy = ByrefLayoutInfo->getType();
    DesTy = DesTy->getPointerTo();
    llvm::Value *BC = Builder.CreatePointerCast(ByrefInfoAddr, DesTy);
    Builder.CreateStore(ByrefLayoutInfo, BC);
  }
}

void CodeGenFunction::BuildBlockRelease(llvm::Value *V, BlockFieldFlags flags) {
  llvm::Value *F = CGM.getBlockObjectDispose();
  llvm::Value *args[] = {
    Builder.CreateBitCast(V, Int8PtrTy),
    llvm::ConstantInt::get(Int32Ty, flags.getBitMask())
  };
  EmitNounwindRuntimeCall(F, args); // FIXME: throwing destructors?
}

namespace {
  struct CallBlockRelease : EHScopeStack::Cleanup {
    llvm::Value *Addr;
    CallBlockRelease(llvm::Value *Addr) : Addr(Addr) {}

    void Emit(CodeGenFunction &CGF, Flags flags) override {
      // Should we be passing FIELD_IS_WEAK here?
      CGF.BuildBlockRelease(Addr, BLOCK_FIELD_IS_BYREF);
    }
  };
}

/// Enter a cleanup to destroy a __block variable.  Note that this
/// cleanup should be a no-op if the variable hasn't left the stack
/// yet; if a cleanup is required for the variable itself, that needs
/// to be done externally.
void CodeGenFunction::enterByrefCleanup(const AutoVarEmission &emission) {
  // We don't enter this cleanup if we're in pure-GC mode.
  if (CGM.getLangOpts().getGC() == LangOptions::GCOnly)
    return;

  EHStack.pushCleanup<CallBlockRelease>(NormalAndEHCleanup, emission.Address);
}

/// Adjust the declaration of something from the blocks API.
static void configureBlocksRuntimeObject(CodeGenModule &CGM,
                                         llvm::Constant *C) {
  if (!CGM.getLangOpts().BlocksRuntimeOptional) return;

  auto *GV = cast<llvm::GlobalValue>(C->stripPointerCasts());
  if (GV->isDeclaration() && GV->hasExternalLinkage())
    GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);
}

llvm::Constant *CodeGenModule::getBlockObjectDispose() {
  if (BlockObjectDispose)
    return BlockObjectDispose;

  llvm::Type *args[] = { Int8PtrTy, Int32Ty };
  llvm::FunctionType *fty
    = llvm::FunctionType::get(VoidTy, args, false);
  BlockObjectDispose = CreateRuntimeFunction(fty, "_Block_object_dispose");
  configureBlocksRuntimeObject(*this, BlockObjectDispose);
  return BlockObjectDispose;
}

llvm::Constant *CodeGenModule::getBlockObjectAssign() {
  if (BlockObjectAssign)
    return BlockObjectAssign;

  llvm::Type *args[] = { Int8PtrTy, Int8PtrTy, Int32Ty };
  llvm::FunctionType *fty
    = llvm::FunctionType::get(VoidTy, args, false);
  BlockObjectAssign = CreateRuntimeFunction(fty, "_Block_object_assign");
  configureBlocksRuntimeObject(*this, BlockObjectAssign);
  return BlockObjectAssign;
}

llvm::Constant *CodeGenModule::getNSConcreteGlobalBlock() {
  if (NSConcreteGlobalBlock)
    return NSConcreteGlobalBlock;

  NSConcreteGlobalBlock = GetOrCreateLLVMGlobal("_NSConcreteGlobalBlock",
                                                Int8PtrTy->getPointerTo(),
                                                nullptr);
  configureBlocksRuntimeObject(*this, NSConcreteGlobalBlock);
  return NSConcreteGlobalBlock;
}

llvm::Constant *CodeGenModule::getNSConcreteStackBlock() {
  if (NSConcreteStackBlock)
    return NSConcreteStackBlock;

  NSConcreteStackBlock = GetOrCreateLLVMGlobal("_NSConcreteStackBlock",
                                               Int8PtrTy->getPointerTo(),
                                               nullptr);
  configureBlocksRuntimeObject(*this, NSConcreteStackBlock);
  return NSConcreteStackBlock;  
}
