//===--- CGRecordLayoutBuilder.cpp - Record builder helper ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a helper class used to build CGRecordLayout objects and LLVM types.
//
//===----------------------------------------------------------------------===//

#include "CGRecordLayoutBuilder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "CodeGenTypes.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetData.h"


using namespace clang;
using namespace CodeGen;

void CGRecordLayoutBuilder::Layout(const RecordDecl *D) {
  Alignment = Types.getContext().getASTRecordLayout(D).getAlignment() / 8;

  if (D->isUnion()) {
    LayoutUnion(D);
    return;
  }

  Packed = D->hasAttr<PackedAttr>();
  
  if (LayoutFields(D))
    return;
  
  // We weren't able to layout the struct. Try again with a packed struct
  Packed = true;
  AlignmentAsLLVMStruct = 1;
  NextFieldOffsetInBytes = 0;
  FieldTypes.clear();
  LLVMFields.clear();
  LLVMBitFields.clear();
  
  LayoutFields(D);
}

void CGRecordLayoutBuilder::LayoutBitField(const FieldDecl *D,
                                           uint64_t FieldOffset) {
  uint64_t FieldSize = 
    D->getBitWidth()->EvaluateAsInt(Types.getContext()).getZExtValue();
  
  if (FieldSize == 0)
    return;

  uint64_t NextFieldOffset = NextFieldOffsetInBytes * 8;
  unsigned NumBytesToAppend;
  
  if (FieldOffset < NextFieldOffset) {
    assert(BitsAvailableInLastField && "Bitfield size mismatch!");
    assert(NextFieldOffsetInBytes && "Must have laid out at least one byte!");
    
    // The bitfield begins in the previous bit-field.
    NumBytesToAppend = 
      llvm::RoundUpToAlignment(FieldSize - BitsAvailableInLastField, 8) / 8;
  } else {
    assert(FieldOffset % 8 == 0 && "Field offset not aligned correctly");

    // Append padding if necessary.
    AppendBytes((FieldOffset - NextFieldOffset) / 8);
    
    NumBytesToAppend = 
      llvm::RoundUpToAlignment(FieldSize, 8) / 8;
    
    assert(NumBytesToAppend && "No bytes to append!");
  }

  const llvm::Type *Ty = Types.ConvertTypeForMemRecursive(D->getType());
  uint64_t TypeSizeInBits = getTypeSizeInBytes(Ty) * 8;
  
  LLVMBitFields.push_back(LLVMBitFieldInfo(D, FieldOffset / TypeSizeInBits,
                                           FieldOffset % TypeSizeInBits, 
                                           FieldSize));
  
  AppendBytes(NumBytesToAppend);
  
  AlignmentAsLLVMStruct = std::max(AlignmentAsLLVMStruct, getTypeAlignment(Ty));

  BitsAvailableInLastField = 
    NextFieldOffsetInBytes * 8 - (FieldOffset + FieldSize);
}

bool CGRecordLayoutBuilder::LayoutField(const FieldDecl *D,
                                        uint64_t FieldOffset) {
  // If the field is packed, then we need a packed struct.
  if (!Packed && D->hasAttr<PackedAttr>())
    return false;

  if (D->isBitField()) {
    // We must use packed structs for unnamed bit fields since they
    // don't affect the struct alignment.
    if (!Packed && !D->getDeclName())
      return false;
    
    LayoutBitField(D, FieldOffset);
    return true;
  }
  
  assert(FieldOffset % 8 == 0 && "FieldOffset is not on a byte boundary!");
  uint64_t FieldOffsetInBytes = FieldOffset / 8;
  
  const llvm::Type *Ty = Types.ConvertTypeForMemRecursive(D->getType());
  unsigned TypeAlignment = getTypeAlignment(Ty);

  // If the type alignment is larger then the struct alignment, we must use
  // a packed struct.
  if (TypeAlignment > Alignment) {
    assert(!Packed && "Alignment is wrong even with packed struct!");
    return false;
  }
  
  if (const RecordType *RT = D->getType()->getAs<RecordType>()) {
    const RecordDecl *RD = cast<RecordDecl>(RT->getDecl());
    if (const PragmaPackAttr *PPA = RD->getAttr<PragmaPackAttr>()) {
      if (PPA->getAlignment() != TypeAlignment * 8 && !Packed)
        return false;
    }
  }

  // Round up the field offset to the alignment of the field type.
  uint64_t AlignedNextFieldOffsetInBytes = 
    llvm::RoundUpToAlignment(NextFieldOffsetInBytes, TypeAlignment);

  if (FieldOffsetInBytes < AlignedNextFieldOffsetInBytes) {
    assert(!Packed && "Could not place field even with packed struct!");
    return false;
  }
  
  if (AlignedNextFieldOffsetInBytes < FieldOffsetInBytes) {
    // Even with alignment, the field offset is not at the right place,
    // insert padding.
    uint64_t PaddingInBytes = FieldOffsetInBytes - NextFieldOffsetInBytes;
    
    AppendBytes(PaddingInBytes);
  }
  
  // Now append the field.
  LLVMFields.push_back(LLVMFieldInfo(D, FieldTypes.size()));
  AppendField(FieldOffsetInBytes, Ty);
  
  return true;
}

void CGRecordLayoutBuilder::LayoutUnion(const RecordDecl *D) {
  assert(D->isUnion() && "Can't call LayoutUnion on a non-union record!");
  
  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);
  
  const llvm::Type *Ty = 0;
  uint64_t Size = 0;
  unsigned Align = 0;
  
  unsigned FieldNo = 0;
  for (RecordDecl::field_iterator Field = D->field_begin(), 
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    assert(Layout.getFieldOffset(FieldNo) == 0 && 
          "Union field offset did not start at the beginning of record!");

    if (Field->isBitField()) {
      uint64_t FieldSize = 
        Field->getBitWidth()->EvaluateAsInt(Types.getContext()).getZExtValue();
    
      // Ignore zero sized bit fields.
      if (FieldSize == 0)
        continue;
      
      // Add the bit field info.
      Types.addBitFieldInfo(*Field, 0, 0, FieldSize);
    } else
      Types.addFieldInfo(*Field, 0);
    
    const llvm::Type *FieldTy = 
      Types.ConvertTypeForMemRecursive(Field->getType());
    unsigned FieldAlign = Types.getTargetData().getABITypeAlignment(FieldTy);
    uint64_t FieldSize = Types.getTargetData().getTypeAllocSize(FieldTy);
    
    if (FieldAlign < Align)
      continue;
    
    if (FieldAlign > Align || FieldSize > Size) {
      Ty = FieldTy;
      Align = FieldAlign;
      Size = FieldSize;
    }
  }
  
  // Now add our field.
  if (Ty) {
    AppendField(0, Ty);

    if (getTypeAlignment(Ty) > Layout.getAlignment() / 8) {
      // We need a packed struct.
      Packed = true;
      Align = 1;
    }
  }
  
  // Append tail padding.
  if (Layout.getSize() / 8 > Size)
    AppendPadding(Layout.getSize() / 8, Align);
}

bool CGRecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  assert(!D->isUnion() && "Can't call LayoutFields on a union!");
  assert(Alignment && "Did not set alignment!");
  
  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);
  
  unsigned FieldNo = 0;

  for (RecordDecl::field_iterator Field = D->field_begin(), 
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    if (!LayoutField(*Field, Layout.getFieldOffset(FieldNo))) {
      assert(!Packed && 
             "Could not layout fields even with a packed LLVM struct!");
      return false;
    }
  }

  // Append tail padding if necessary.
  AppendTailPadding(Layout.getSize());
  
  return true;
}

void CGRecordLayoutBuilder::AppendTailPadding(uint64_t RecordSize) {
  assert(RecordSize % 8 == 0 && "Invalid record size!");
  
  uint64_t RecordSizeInBytes = RecordSize / 8;
  assert(NextFieldOffsetInBytes <= RecordSizeInBytes && "Size mismatch!");
  
  unsigned NumPadBytes = RecordSizeInBytes - NextFieldOffsetInBytes;
  AppendBytes(NumPadBytes);
}

void CGRecordLayoutBuilder::AppendField(uint64_t FieldOffsetInBytes, 
                                        const llvm::Type *FieldTy) {
  AlignmentAsLLVMStruct = std::max(AlignmentAsLLVMStruct,
                                   getTypeAlignment(FieldTy));

  uint64_t FieldSizeInBytes = getTypeSizeInBytes(FieldTy);

  FieldTypes.push_back(FieldTy);

  NextFieldOffsetInBytes = FieldOffsetInBytes + FieldSizeInBytes;
  BitsAvailableInLastField = 0;
}

void 
CGRecordLayoutBuilder::AppendPadding(uint64_t FieldOffsetInBytes,
                                     const llvm::Type *FieldTy) {
  AppendPadding(FieldOffsetInBytes, getTypeAlignment(FieldTy));
}

void CGRecordLayoutBuilder::AppendPadding(uint64_t FieldOffsetInBytes, 
                                          unsigned FieldAlignment) {
  assert(NextFieldOffsetInBytes <= FieldOffsetInBytes &&
         "Incorrect field layout!");
  
  // Round up the field offset to the alignment of the field type.
  uint64_t AlignedNextFieldOffsetInBytes = 
    llvm::RoundUpToAlignment(NextFieldOffsetInBytes, FieldAlignment);

  if (AlignedNextFieldOffsetInBytes < FieldOffsetInBytes) {
    // Even with alignment, the field offset is not at the right place,
    // insert padding.
    uint64_t PaddingInBytes = FieldOffsetInBytes - NextFieldOffsetInBytes;

    AppendBytes(PaddingInBytes);
  }
}

void CGRecordLayoutBuilder::AppendBytes(uint64_t NumBytes) {
  if (NumBytes == 0)
    return;
  
  const llvm::Type *Ty = llvm::Type::getInt8Ty(Types.getLLVMContext());
  if (NumBytes > 1)
    Ty = llvm::ArrayType::get(Ty, NumBytes);
  
  // Append the padding field
  AppendField(NextFieldOffsetInBytes, Ty);
}

unsigned CGRecordLayoutBuilder::getTypeAlignment(const llvm::Type *Ty) const {
  if (Packed)
    return 1;
  
  return Types.getTargetData().getABITypeAlignment(Ty);
}

uint64_t CGRecordLayoutBuilder::getTypeSizeInBytes(const llvm::Type *Ty) const {
  return Types.getTargetData().getTypeAllocSize(Ty);
}

CGRecordLayout *
CGRecordLayoutBuilder::ComputeLayout(CodeGenTypes &Types,
                                     const RecordDecl *D) {
  CGRecordLayoutBuilder Builder(Types);
  
  Builder.Layout(D);

  const llvm::Type *Ty = llvm::StructType::get(Types.getLLVMContext(),
                                               Builder.FieldTypes,
                                               Builder.Packed);
  assert(Types.getContext().getASTRecordLayout(D).getSize() / 8 ==
         Types.getTargetData().getTypeAllocSize(Ty) &&
         "Type size mismatch!");
  
  // Add all the field numbers.
  for (unsigned i = 0, e = Builder.LLVMFields.size(); i != e; ++i) {
    const FieldDecl *FD = Builder.LLVMFields[i].first;
    unsigned FieldNo = Builder.LLVMFields[i].second;

    Types.addFieldInfo(FD, FieldNo);
  }

  // Add bitfield info.
  for (unsigned i = 0, e = Builder.LLVMBitFields.size(); i != e; ++i) {
    const LLVMBitFieldInfo &Info = Builder.LLVMBitFields[i];
    
    Types.addBitFieldInfo(Info.FD, Info.FieldNo, Info.Start, Info.Size);
  }
  
  return new CGRecordLayout(Ty, llvm::SmallSet<unsigned, 8>());
}
