//===--  BitcodeReader.cpp - ClangDoc Bitcode Reader ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BitcodeReader.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace doc {

using Record = llvm::SmallVector<uint64_t, 1024>;

llvm::Error decodeRecord(Record R, llvm::SmallVectorImpl<char> &Field,
                         llvm::StringRef Blob) {
  Field.assign(Blob.begin(), Blob.end());
  return llvm::Error::success();
}

llvm::Error decodeRecord(Record R, SymbolID &Field, llvm::StringRef Blob) {
  if (R[0] != BitCodeConstants::USRHashSize)
    return llvm::make_error<llvm::StringError>("Incorrect USR size.\n",
                                               llvm::inconvertibleErrorCode());

  // First position in the record is the length of the following array, so we
  // copy the following elements to the field.
  for (int I = 0, E = R[0]; I < E; ++I)
    Field[I] = R[I + 1];
  return llvm::Error::success();
}

llvm::Error decodeRecord(Record R, bool &Field, llvm::StringRef Blob) {
  Field = R[0] != 0;
  return llvm::Error::success();
}

llvm::Error decodeRecord(Record R, int &Field, llvm::StringRef Blob) {
  if (R[0] > INT_MAX)
    return llvm::make_error<llvm::StringError>("Integer too large to parse.\n",
                                               llvm::inconvertibleErrorCode());
  Field = (int)R[0];
  return llvm::Error::success();
}

llvm::Error decodeRecord(Record R, AccessSpecifier &Field,
                         llvm::StringRef Blob) {
  switch (R[0]) {
  case AS_public:
  case AS_private:
  case AS_protected:
  case AS_none:
    Field = (AccessSpecifier)R[0];
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid value for AccessSpecifier.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error decodeRecord(Record R, TagTypeKind &Field, llvm::StringRef Blob) {
  switch (R[0]) {
  case TTK_Struct:
  case TTK_Interface:
  case TTK_Union:
  case TTK_Class:
  case TTK_Enum:
    Field = (TagTypeKind)R[0];
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid value for TagTypeKind.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error decodeRecord(Record R, llvm::Optional<Location> &Field,
                         llvm::StringRef Blob) {
  if (R[0] > INT_MAX)
    return llvm::make_error<llvm::StringError>("Integer too large to parse.\n",
                                               llvm::inconvertibleErrorCode());
  Field.emplace((int)R[0], Blob);
  return llvm::Error::success();
}

llvm::Error decodeRecord(Record R, InfoType &Field, llvm::StringRef Blob) {
  switch (auto IT = static_cast<InfoType>(R[0])) {
  case InfoType::IT_namespace:
  case InfoType::IT_record:
  case InfoType::IT_function:
  case InfoType::IT_default:
  case InfoType::IT_enum:
    Field = IT;
    return llvm::Error::success();
  }
  return llvm::make_error<llvm::StringError>("Invalid value for InfoType.\n",
                                             llvm::inconvertibleErrorCode());
}

llvm::Error decodeRecord(Record R, FieldId &Field, llvm::StringRef Blob) {
  switch (auto F = static_cast<FieldId>(R[0])) {
  case FieldId::F_namespace:
  case FieldId::F_parent:
  case FieldId::F_vparent:
  case FieldId::F_type:
  case FieldId::F_child_namespace:
  case FieldId::F_child_record:
  case FieldId::F_default:
    Field = F;
    return llvm::Error::success();
  }
  return llvm::make_error<llvm::StringError>("Invalid value for FieldId.\n",
                                             llvm::inconvertibleErrorCode());
}

llvm::Error decodeRecord(Record R,
                         llvm::SmallVectorImpl<llvm::SmallString<16>> &Field,
                         llvm::StringRef Blob) {
  Field.push_back(Blob);
  return llvm::Error::success();
}

llvm::Error decodeRecord(Record R, llvm::SmallVectorImpl<Location> &Field,
                         llvm::StringRef Blob) {
  if (R[0] > INT_MAX)
    return llvm::make_error<llvm::StringError>("Integer too large to parse.\n",
                                               llvm::inconvertibleErrorCode());
  Field.emplace_back((int)R[0], Blob);
  return llvm::Error::success();
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        const unsigned VersionNo) {
  if (ID == VERSION && R[0] == VersionNo)
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(
      "Mismatched bitcode version number.\n", llvm::inconvertibleErrorCode());
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        NamespaceInfo *I) {
  switch (ID) {
  case NAMESPACE_USR:
    return decodeRecord(R, I->USR, Blob);
  case NAMESPACE_NAME:
    return decodeRecord(R, I->Name, Blob);
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid field for NamespaceInfo.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        RecordInfo *I) {
  switch (ID) {
  case RECORD_USR:
    return decodeRecord(R, I->USR, Blob);
  case RECORD_NAME:
    return decodeRecord(R, I->Name, Blob);
  case RECORD_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case RECORD_LOCATION:
    return decodeRecord(R, I->Loc, Blob);
  case RECORD_TAG_TYPE:
    return decodeRecord(R, I->TagType, Blob);
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid field for RecordInfo.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        EnumInfo *I) {
  switch (ID) {
  case ENUM_USR:
    return decodeRecord(R, I->USR, Blob);
  case ENUM_NAME:
    return decodeRecord(R, I->Name, Blob);
  case ENUM_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case ENUM_LOCATION:
    return decodeRecord(R, I->Loc, Blob);
  case ENUM_MEMBER:
    return decodeRecord(R, I->Members, Blob);
  case ENUM_SCOPED:
    return decodeRecord(R, I->Scoped, Blob);
  default:
    return llvm::make_error<llvm::StringError>("Invalid field for EnumInfo.\n",
                                               llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        FunctionInfo *I) {
  switch (ID) {
  case FUNCTION_USR:
    return decodeRecord(R, I->USR, Blob);
  case FUNCTION_NAME:
    return decodeRecord(R, I->Name, Blob);
  case FUNCTION_DEFLOCATION:
    return decodeRecord(R, I->DefLoc, Blob);
  case FUNCTION_LOCATION:
    return decodeRecord(R, I->Loc, Blob);
  case FUNCTION_ACCESS:
    return decodeRecord(R, I->Access, Blob);
  case FUNCTION_IS_METHOD:
    return decodeRecord(R, I->IsMethod, Blob);
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid field for FunctionInfo.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        TypeInfo *I) {
  return llvm::Error::success();
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        FieldTypeInfo *I) {
  switch (ID) {
  case FIELD_TYPE_NAME:
    return decodeRecord(R, I->Name, Blob);
  default:
    return llvm::make_error<llvm::StringError>("Invalid field for TypeInfo.\n",
                                               llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        MemberTypeInfo *I) {
  switch (ID) {
  case MEMBER_TYPE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case MEMBER_TYPE_ACCESS:
    return decodeRecord(R, I->Access, Blob);
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid field for MemberTypeInfo.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        CommentInfo *I) {
  switch (ID) {
  case COMMENT_KIND:
    return decodeRecord(R, I->Kind, Blob);
  case COMMENT_TEXT:
    return decodeRecord(R, I->Text, Blob);
  case COMMENT_NAME:
    return decodeRecord(R, I->Name, Blob);
  case COMMENT_DIRECTION:
    return decodeRecord(R, I->Direction, Blob);
  case COMMENT_PARAMNAME:
    return decodeRecord(R, I->ParamName, Blob);
  case COMMENT_CLOSENAME:
    return decodeRecord(R, I->CloseName, Blob);
  case COMMENT_ATTRKEY:
    return decodeRecord(R, I->AttrKeys, Blob);
  case COMMENT_ATTRVAL:
    return decodeRecord(R, I->AttrValues, Blob);
  case COMMENT_ARG:
    return decodeRecord(R, I->Args, Blob);
  case COMMENT_SELFCLOSING:
    return decodeRecord(R, I->SelfClosing, Blob);
  case COMMENT_EXPLICIT:
    return decodeRecord(R, I->Explicit, Blob);
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid field for CommentInfo.\n", llvm::inconvertibleErrorCode());
  }
}

llvm::Error parseRecord(Record R, unsigned ID, llvm::StringRef Blob,
                        Reference *I, FieldId &F) {
  switch (ID) {
  case REFERENCE_USR:
    return decodeRecord(R, I->USR, Blob);
  case REFERENCE_NAME:
    return decodeRecord(R, I->Name, Blob);
  case REFERENCE_TYPE:
    return decodeRecord(R, I->RefType, Blob);
  case REFERENCE_FIELD:
    return decodeRecord(R, F, Blob);
  default:
    return llvm::make_error<llvm::StringError>("Invalid field for Reference.\n",
                                               llvm::inconvertibleErrorCode());
  }
}

template <typename T> llvm::Expected<CommentInfo *> getCommentInfo(T I) {
  return llvm::make_error<llvm::StringError>(
      "Invalid type cannot contain CommentInfo.\n",
      llvm::inconvertibleErrorCode());
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(FunctionInfo *I) {
  I->Description.emplace_back();
  return &I->Description.back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(NamespaceInfo *I) {
  I->Description.emplace_back();
  return &I->Description.back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(RecordInfo *I) {
  I->Description.emplace_back();
  return &I->Description.back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(EnumInfo *I) {
  I->Description.emplace_back();
  return &I->Description.back();
}

template <> llvm::Expected<CommentInfo *> getCommentInfo(CommentInfo *I) {
  I->Children.emplace_back(llvm::make_unique<CommentInfo>());
  return I->Children.back().get();
}

template <>
llvm::Expected<CommentInfo *> getCommentInfo(std::unique_ptr<CommentInfo> &I) {
  return getCommentInfo(I.get());
}

template <typename T, typename TTypeInfo>
llvm::Error addTypeInfo(T I, TTypeInfo &&TI) {
  return llvm::make_error<llvm::StringError>(
      "Invalid type cannot contain TypeInfo.\n",
      llvm::inconvertibleErrorCode());
}

template <> llvm::Error addTypeInfo(RecordInfo *I, MemberTypeInfo &&T) {
  I->Members.emplace_back(std::move(T));
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(FunctionInfo *I, TypeInfo &&T) {
  I->ReturnType = std::move(T);
  return llvm::Error::success();
}

template <> llvm::Error addTypeInfo(FunctionInfo *I, FieldTypeInfo &&T) {
  I->Params.emplace_back(std::move(T));
  return llvm::Error::success();
}

template <typename T> llvm::Error addReference(T I, Reference &&R, FieldId F) {
  return llvm::make_error<llvm::StringError>(
      "Invalid type cannot contain Reference\n",
      llvm::inconvertibleErrorCode());
}

template <> llvm::Error addReference(TypeInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_type:
    I->Type = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <>
llvm::Error addReference(FieldTypeInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_type:
    I->Type = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <>
llvm::Error addReference(MemberTypeInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_type:
    I->Type = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <> llvm::Error addReference(EnumInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <>
llvm::Error addReference(NamespaceInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_child_namespace:
    I->ChildNamespaces.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_child_record:
    I->ChildRecords.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <>
llvm::Error addReference(FunctionInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_parent:
    I->Parent = std::move(R);
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <> llvm::Error addReference(RecordInfo *I, Reference &&R, FieldId F) {
  switch (F) {
  case FieldId::F_namespace:
    I->Namespace.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_parent:
    I->Parents.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_vparent:
    I->VirtualParents.emplace_back(std::move(R));
    return llvm::Error::success();
  case FieldId::F_child_record:
    I->ChildRecords.emplace_back(std::move(R));
    return llvm::Error::success();
  default:
    return llvm::make_error<llvm::StringError>(
        "Invalid type cannot contain Reference.\n",
        llvm::inconvertibleErrorCode());
  }
}

template <typename T, typename ChildInfoType>
void addChild(T I, ChildInfoType &&R) {
  llvm::errs() << "Invalid child type for info.\n";
  exit(1);
}

template <> void addChild(NamespaceInfo *I, FunctionInfo &&R) {
  I->ChildFunctions.emplace_back(std::move(R));
}

template <> void addChild(NamespaceInfo *I, EnumInfo &&R) {
  I->ChildEnums.emplace_back(std::move(R));
}

template <> void addChild(RecordInfo *I, FunctionInfo &&R) {
  I->ChildFunctions.emplace_back(std::move(R));
}

template <> void addChild(RecordInfo *I, EnumInfo &&R) {
  I->ChildEnums.emplace_back(std::move(R));
}

// Read records from bitcode into a given info.
template <typename T>
llvm::Error ClangDocBitcodeReader::readRecord(unsigned ID, T I) {
  Record R;
  llvm::StringRef Blob;
  unsigned RecID = Stream.readRecord(ID, R, &Blob);
  return parseRecord(R, RecID, Blob, I);
}

template <>
llvm::Error ClangDocBitcodeReader::readRecord(unsigned ID, Reference *I) {
  Record R;
  llvm::StringRef Blob;
  unsigned RecID = Stream.readRecord(ID, R, &Blob);
  return parseRecord(R, RecID, Blob, I, CurrentReferenceField);
}

// Read a block of records into a single info.
template <typename T>
llvm::Error ClangDocBitcodeReader::readBlock(unsigned ID, T I) {
  if (Stream.EnterSubBlock(ID))
    return llvm::make_error<llvm::StringError>("Unable to enter subblock.\n",
                                               llvm::inconvertibleErrorCode());

  while (true) {
    unsigned BlockOrCode = 0;
    Cursor Res = skipUntilRecordOrBlock(BlockOrCode);

    switch (Res) {
    case Cursor::BadBlock:
      return llvm::make_error<llvm::StringError>(
          "Bad block found.\n", llvm::inconvertibleErrorCode());
    case Cursor::BlockEnd:
      return llvm::Error::success();
    case Cursor::BlockBegin:
      if (auto Err = readSubBlock(BlockOrCode, I)) {
        if (!Stream.SkipBlock())
          continue;
        return Err;
      }
      continue;
    case Cursor::Record:
      break;
    }
    if (auto Err = readRecord(BlockOrCode, I))
      return Err;
  }
}

template <typename T>
llvm::Error ClangDocBitcodeReader::readSubBlock(unsigned ID, T I) {
  switch (ID) {
  // Blocks can only have Comment, Reference, TypeInfo, FunctionInfo, or
  // EnumInfo subblocks
  case BI_COMMENT_BLOCK_ID: {
    auto Comment = getCommentInfo(I);
    if (!Comment)
      return Comment.takeError();
    if (auto Err = readBlock(ID, Comment.get()))
      return Err;
    return llvm::Error::success();
  }
  case BI_TYPE_BLOCK_ID: {
    TypeInfo TI;
    if (auto Err = readBlock(ID, &TI))
      return Err;
    if (auto Err = addTypeInfo(I, std::move(TI)))
      return Err;
    return llvm::Error::success();
  }
  case BI_FIELD_TYPE_BLOCK_ID: {
    FieldTypeInfo TI;
    if (auto Err = readBlock(ID, &TI))
      return Err;
    if (auto Err = addTypeInfo(I, std::move(TI)))
      return Err;
    return llvm::Error::success();
  }
  case BI_MEMBER_TYPE_BLOCK_ID: {
    MemberTypeInfo TI;
    if (auto Err = readBlock(ID, &TI))
      return Err;
    if (auto Err = addTypeInfo(I, std::move(TI)))
      return Err;
    return llvm::Error::success();
  }
  case BI_REFERENCE_BLOCK_ID: {
    Reference R;
    if (auto Err = readBlock(ID, &R))
      return Err;
    if (auto Err = addReference(I, std::move(R), CurrentReferenceField))
      return Err;
    return llvm::Error::success();
  }
  case BI_FUNCTION_BLOCK_ID: {
    FunctionInfo F;
    if (auto Err = readBlock(ID, &F))
      return Err;
    addChild(I, std::move(F));
    return llvm::Error::success();
  }
  case BI_ENUM_BLOCK_ID: {
    EnumInfo E;
    if (auto Err = readBlock(ID, &E))
      return Err;
    addChild(I, std::move(E));
    return llvm::Error::success();
  }
  default:
    return llvm::make_error<llvm::StringError>("Invalid subblock type.\n",
                                               llvm::inconvertibleErrorCode());
  }
}

ClangDocBitcodeReader::Cursor
ClangDocBitcodeReader::skipUntilRecordOrBlock(unsigned &BlockOrRecordID) {
  BlockOrRecordID = 0;

  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    switch ((llvm::bitc::FixedAbbrevIDs)Code) {
    case llvm::bitc::ENTER_SUBBLOCK:
      BlockOrRecordID = Stream.ReadSubBlockID();
      return Cursor::BlockBegin;
    case llvm::bitc::END_BLOCK:
      if (Stream.ReadBlockEnd())
        return Cursor::BadBlock;
      return Cursor::BlockEnd;
    case llvm::bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      continue;
    case llvm::bitc::UNABBREV_RECORD:
      return Cursor::BadBlock;
    default:
      BlockOrRecordID = Code;
      return Cursor::Record;
    }
  }
  llvm_unreachable("Premature stream end.");
}

llvm::Error ClangDocBitcodeReader::validateStream() {
  if (Stream.AtEndOfStream())
    return llvm::make_error<llvm::StringError>("Premature end of stream.\n",
                                               llvm::inconvertibleErrorCode());

  // Sniff for the signature.
  if (Stream.Read(8) != BitCodeConstants::Signature[0] ||
      Stream.Read(8) != BitCodeConstants::Signature[1] ||
      Stream.Read(8) != BitCodeConstants::Signature[2] ||
      Stream.Read(8) != BitCodeConstants::Signature[3])
    return llvm::make_error<llvm::StringError>("Invalid bitcode signature.\n",
                                               llvm::inconvertibleErrorCode());
  return llvm::Error::success();
}

llvm::Error ClangDocBitcodeReader::readBlockInfoBlock() {
  BlockInfo = Stream.ReadBlockInfoBlock();
  if (!BlockInfo)
    return llvm::make_error<llvm::StringError>(
        "Unable to parse BlockInfoBlock.\n", llvm::inconvertibleErrorCode());
  Stream.setBlockInfo(&*BlockInfo);
  return llvm::Error::success();
}

template <typename T>
llvm::Expected<std::unique_ptr<Info>>
ClangDocBitcodeReader::createInfo(unsigned ID) {
  std::unique_ptr<Info> I = llvm::make_unique<T>();
  if (auto Err = readBlock(ID, static_cast<T *>(I.get())))
    return std::move(Err);
  return std::unique_ptr<Info>{std::move(I)};;
}

llvm::Expected<std::unique_ptr<Info>>
ClangDocBitcodeReader::readBlockToInfo(unsigned ID) {
  switch (ID) {
  case BI_NAMESPACE_BLOCK_ID:
    return createInfo<NamespaceInfo>(ID);
  case BI_RECORD_BLOCK_ID:
    return createInfo<RecordInfo>(ID);
  case BI_ENUM_BLOCK_ID:
    return createInfo<EnumInfo>(ID);
  case BI_FUNCTION_BLOCK_ID:
    return createInfo<FunctionInfo>(ID);
  default:
    return llvm::make_error<llvm::StringError>("Cannot create info.\n",
                                               llvm::inconvertibleErrorCode());
  }
}

// Entry point
llvm::Expected<std::vector<std::unique_ptr<Info>>>
ClangDocBitcodeReader::readBitcode() {
  std::vector<std::unique_ptr<Info>> Infos;
  if (auto Err = validateStream())
    return std::move(Err);

  // Read the top level blocks.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code != llvm::bitc::ENTER_SUBBLOCK)
      return llvm::make_error<llvm::StringError>(
          "No blocks in input.\n", llvm::inconvertibleErrorCode());
    unsigned ID = Stream.ReadSubBlockID();
    switch (ID) {
    // NamedType and Comment blocks should not appear at the top level
    case BI_TYPE_BLOCK_ID:
    case BI_FIELD_TYPE_BLOCK_ID:
    case BI_MEMBER_TYPE_BLOCK_ID:
    case BI_COMMENT_BLOCK_ID:
    case BI_REFERENCE_BLOCK_ID:
      return llvm::make_error<llvm::StringError>(
          "Invalid top level block.\n", llvm::inconvertibleErrorCode());
    case BI_NAMESPACE_BLOCK_ID:
    case BI_RECORD_BLOCK_ID:
    case BI_ENUM_BLOCK_ID:
    case BI_FUNCTION_BLOCK_ID: {
      auto InfoOrErr = readBlockToInfo(ID);
      if (!InfoOrErr)
        return InfoOrErr.takeError();
      Infos.emplace_back(std::move(InfoOrErr.get()));
      continue;
    }
    case BI_VERSION_BLOCK_ID:
      if (auto Err = readBlock(ID, VersionNumber))
        return std::move(Err);
      continue;
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (auto Err = readBlockInfoBlock())
        return std::move(Err);
      continue;
    default:
      if (!Stream.SkipBlock())
        continue;
    }
  }
  return std::move(Infos);
}

} // namespace doc
} // namespace clang
