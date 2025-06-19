//===--  BitcodeWriter.cpp - ClangDoc Bitcode Writer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BitcodeWriter.h"
#include "llvm/ADT/IndexedMap.h"
#include <initializer_list>

namespace clang {
namespace doc {

// Empty SymbolID for comparison, so we don't have to construct one every time.
static const SymbolID EmptySID = SymbolID();

// Since id enums are not zero-indexed, we need to transform the given id into
// its associated index.
struct BlockIdToIndexFunctor {
  using argument_type = unsigned;
  unsigned operator()(unsigned ID) const { return ID - BI_FIRST; }
};

struct RecordIdToIndexFunctor {
  using argument_type = unsigned;
  unsigned operator()(unsigned ID) const { return ID - RI_FIRST; }
};

using AbbrevDsc = void (*)(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev);

static void
generateAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev,
               const std::initializer_list<llvm::BitCodeAbbrevOp> Ops) {
  for (const auto &Op : Ops)
    Abbrev->Add(Op);
}

static void genBoolAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  generateAbbrev(Abbrev,
                 {// 0. Boolean
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                        BitCodeConstants::BoolSize)});
}

static void genIntAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  generateAbbrev(Abbrev,
                 {// 0. Fixed-size integer
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                        BitCodeConstants::IntSize)});
}

static void genSymbolIdAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  generateAbbrev(Abbrev,
                 {// 0. Fixed-size integer (length of the sha1'd USR)
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                        BitCodeConstants::USRLengthSize),
                  // 1. Fixed-size array of Char6 (USR)
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array),
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                        BitCodeConstants::USRBitLengthSize)});
}

static void genStringAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  generateAbbrev(Abbrev,
                 {// 0. Fixed-size integer (length of the following string)
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                        BitCodeConstants::StringLengthSize),
                  // 1. The string blob
                  llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Blob)});
}

// Assumes that the file will not have more than 65535 lines.
static void genLocationAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  generateAbbrev(
      Abbrev,
      {// 0. Fixed-size integer (line number)
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                             BitCodeConstants::LineNumberSize),
       // 1. Fixed-size integer (start line number)
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                             BitCodeConstants::LineNumberSize),
       // 2. Boolean (IsFileInRootDir)
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                             BitCodeConstants::BoolSize),
       // 3. Fixed-size integer (length of the following string (filename))
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                             BitCodeConstants::StringLengthSize),
       // 4. The string blob
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Blob)});
}

struct RecordIdDsc {
  llvm::StringRef Name;
  AbbrevDsc Abbrev = nullptr;

  RecordIdDsc() = default;
  RecordIdDsc(llvm::StringRef Name, AbbrevDsc Abbrev)
      : Name(Name), Abbrev(Abbrev) {}

  // Is this 'description' valid?
  operator bool() const {
    return Abbrev != nullptr && Name.data() != nullptr && !Name.empty();
  }
};

static const llvm::IndexedMap<llvm::StringRef, BlockIdToIndexFunctor>
    BlockIdNameMap = []() {
      llvm::IndexedMap<llvm::StringRef, BlockIdToIndexFunctor> BlockIdNameMap;
      BlockIdNameMap.resize(BlockIdCount);

      // There is no init-list constructor for the IndexedMap, so have to
      // improvise
      static const std::vector<std::pair<BlockId, const char *const>> Inits = {
          {BI_VERSION_BLOCK_ID, "VersionBlock"},
          {BI_NAMESPACE_BLOCK_ID, "NamespaceBlock"},
          {BI_ENUM_BLOCK_ID, "EnumBlock"},
          {BI_ENUM_VALUE_BLOCK_ID, "EnumValueBlock"},
          {BI_TYPEDEF_BLOCK_ID, "TypedefBlock"},
          {BI_TYPE_BLOCK_ID, "TypeBlock"},
          {BI_FIELD_TYPE_BLOCK_ID, "FieldTypeBlock"},
          {BI_MEMBER_TYPE_BLOCK_ID, "MemberTypeBlock"},
          {BI_RECORD_BLOCK_ID, "RecordBlock"},
          {BI_BASE_RECORD_BLOCK_ID, "BaseRecordBlock"},
          {BI_FUNCTION_BLOCK_ID, "FunctionBlock"},
          {BI_COMMENT_BLOCK_ID, "CommentBlock"},
          {BI_REFERENCE_BLOCK_ID, "ReferenceBlock"},
          {BI_TEMPLATE_BLOCK_ID, "TemplateBlock"},
          {BI_TEMPLATE_SPECIALIZATION_BLOCK_ID, "TemplateSpecializationBlock"},
          {BI_TEMPLATE_PARAM_BLOCK_ID, "TemplateParamBlock"}};
      assert(Inits.size() == BlockIdCount);
      for (const auto &Init : Inits)
        BlockIdNameMap[Init.first] = Init.second;
      assert(BlockIdNameMap.size() == BlockIdCount);
      return BlockIdNameMap;
    }();

static const llvm::IndexedMap<RecordIdDsc, RecordIdToIndexFunctor>
    RecordIdNameMap = []() {
      llvm::IndexedMap<RecordIdDsc, RecordIdToIndexFunctor> RecordIdNameMap;
      RecordIdNameMap.resize(RecordIdCount);

      // There is no init-list constructor for the IndexedMap, so have to
      // improvise
      static const std::vector<std::pair<RecordId, RecordIdDsc>> Inits = {
          {VERSION, {"Version", &genIntAbbrev}},
          {COMMENT_KIND, {"Kind", &genStringAbbrev}},
          {COMMENT_TEXT, {"Text", &genStringAbbrev}},
          {COMMENT_NAME, {"Name", &genStringAbbrev}},
          {COMMENT_DIRECTION, {"Direction", &genStringAbbrev}},
          {COMMENT_PARAMNAME, {"ParamName", &genStringAbbrev}},
          {COMMENT_CLOSENAME, {"CloseName", &genStringAbbrev}},
          {COMMENT_SELFCLOSING, {"SelfClosing", &genBoolAbbrev}},
          {COMMENT_EXPLICIT, {"Explicit", &genBoolAbbrev}},
          {COMMENT_ATTRKEY, {"AttrKey", &genStringAbbrev}},
          {COMMENT_ATTRVAL, {"AttrVal", &genStringAbbrev}},
          {COMMENT_ARG, {"Arg", &genStringAbbrev}},
          {FIELD_TYPE_NAME, {"Name", &genStringAbbrev}},
          {FIELD_DEFAULT_VALUE, {"DefaultValue", &genStringAbbrev}},
          {MEMBER_TYPE_NAME, {"Name", &genStringAbbrev}},
          {MEMBER_TYPE_ACCESS, {"Access", &genIntAbbrev}},
          {MEMBER_TYPE_IS_STATIC, {"IsStatic", &genBoolAbbrev}},
          {NAMESPACE_USR, {"USR", &genSymbolIdAbbrev}},
          {NAMESPACE_NAME, {"Name", &genStringAbbrev}},
          {NAMESPACE_PATH, {"Path", &genStringAbbrev}},
          {ENUM_USR, {"USR", &genSymbolIdAbbrev}},
          {ENUM_NAME, {"Name", &genStringAbbrev}},
          {ENUM_DEFLOCATION, {"DefLocation", &genLocationAbbrev}},
          {ENUM_LOCATION, {"Location", &genLocationAbbrev}},
          {ENUM_SCOPED, {"Scoped", &genBoolAbbrev}},
          {ENUM_VALUE_NAME, {"Name", &genStringAbbrev}},
          {ENUM_VALUE_VALUE, {"Value", &genStringAbbrev}},
          {ENUM_VALUE_EXPR, {"Expr", &genStringAbbrev}},
          {RECORD_USR, {"USR", &genSymbolIdAbbrev}},
          {RECORD_NAME, {"Name", &genStringAbbrev}},
          {RECORD_PATH, {"Path", &genStringAbbrev}},
          {RECORD_DEFLOCATION, {"DefLocation", &genLocationAbbrev}},
          {RECORD_LOCATION, {"Location", &genLocationAbbrev}},
          {RECORD_TAG_TYPE, {"TagType", &genIntAbbrev}},
          {RECORD_IS_TYPE_DEF, {"IsTypeDef", &genBoolAbbrev}},
          {BASE_RECORD_USR, {"USR", &genSymbolIdAbbrev}},
          {BASE_RECORD_NAME, {"Name", &genStringAbbrev}},
          {BASE_RECORD_PATH, {"Path", &genStringAbbrev}},
          {BASE_RECORD_TAG_TYPE, {"TagType", &genIntAbbrev}},
          {BASE_RECORD_IS_VIRTUAL, {"IsVirtual", &genBoolAbbrev}},
          {BASE_RECORD_ACCESS, {"Access", &genIntAbbrev}},
          {BASE_RECORD_IS_PARENT, {"IsParent", &genBoolAbbrev}},
          {FUNCTION_USR, {"USR", &genSymbolIdAbbrev}},
          {FUNCTION_NAME, {"Name", &genStringAbbrev}},
          {FUNCTION_DEFLOCATION, {"DefLocation", &genLocationAbbrev}},
          {FUNCTION_LOCATION, {"Location", &genLocationAbbrev}},
          {FUNCTION_ACCESS, {"Access", &genIntAbbrev}},
          {FUNCTION_IS_METHOD, {"IsMethod", &genBoolAbbrev}},
          {FUNCTION_IS_STATIC, {"IsStatic", &genBoolAbbrev}},
          {REFERENCE_USR, {"USR", &genSymbolIdAbbrev}},
          {REFERENCE_NAME, {"Name", &genStringAbbrev}},
          {REFERENCE_QUAL_NAME, {"QualName", &genStringAbbrev}},
          {REFERENCE_TYPE, {"RefType", &genIntAbbrev}},
          {REFERENCE_PATH, {"Path", &genStringAbbrev}},
          {REFERENCE_FIELD, {"Field", &genIntAbbrev}},
          {TEMPLATE_PARAM_CONTENTS, {"Contents", &genStringAbbrev}},
          {TEMPLATE_SPECIALIZATION_OF,
           {"SpecializationOf", &genSymbolIdAbbrev}},
          {TYPEDEF_USR, {"USR", &genSymbolIdAbbrev}},
          {TYPEDEF_NAME, {"Name", &genStringAbbrev}},
          {TYPEDEF_DEFLOCATION, {"DefLocation", &genLocationAbbrev}},
          {TYPEDEF_IS_USING, {"IsUsing", &genBoolAbbrev}}};
      assert(Inits.size() == RecordIdCount);
      for (const auto &Init : Inits) {
        RecordIdNameMap[Init.first] = Init.second;
        assert((Init.second.Name.size() + 1) <= BitCodeConstants::RecordSize);
      }
      assert(RecordIdNameMap.size() == RecordIdCount);
      return RecordIdNameMap;
    }();

static const std::vector<std::pair<BlockId, std::vector<RecordId>>>
    RecordsByBlock{
        // Version Block
        {BI_VERSION_BLOCK_ID, {VERSION}},
        // Comment Block
        {BI_COMMENT_BLOCK_ID,
         {COMMENT_KIND, COMMENT_TEXT, COMMENT_NAME, COMMENT_DIRECTION,
          COMMENT_PARAMNAME, COMMENT_CLOSENAME, COMMENT_SELFCLOSING,
          COMMENT_EXPLICIT, COMMENT_ATTRKEY, COMMENT_ATTRVAL, COMMENT_ARG}},
        // Type Block
        {BI_TYPE_BLOCK_ID, {}},
        // FieldType Block
        {BI_FIELD_TYPE_BLOCK_ID, {FIELD_TYPE_NAME, FIELD_DEFAULT_VALUE}},
        // MemberType Block
        {BI_MEMBER_TYPE_BLOCK_ID,
         {MEMBER_TYPE_NAME, MEMBER_TYPE_ACCESS, MEMBER_TYPE_IS_STATIC}},
        // Enum Block
        {BI_ENUM_BLOCK_ID,
         {ENUM_USR, ENUM_NAME, ENUM_DEFLOCATION, ENUM_LOCATION, ENUM_SCOPED}},
        // Enum Value Block
        {BI_ENUM_VALUE_BLOCK_ID,
         {ENUM_VALUE_NAME, ENUM_VALUE_VALUE, ENUM_VALUE_EXPR}},
        // Typedef Block
        {BI_TYPEDEF_BLOCK_ID,
         {TYPEDEF_USR, TYPEDEF_NAME, TYPEDEF_DEFLOCATION, TYPEDEF_IS_USING}},
        // Namespace Block
        {BI_NAMESPACE_BLOCK_ID,
         {NAMESPACE_USR, NAMESPACE_NAME, NAMESPACE_PATH}},
        // Record Block
        {BI_RECORD_BLOCK_ID,
         {RECORD_USR, RECORD_NAME, RECORD_PATH, RECORD_DEFLOCATION,
          RECORD_LOCATION, RECORD_TAG_TYPE, RECORD_IS_TYPE_DEF}},
        // BaseRecord Block
        {BI_BASE_RECORD_BLOCK_ID,
         {BASE_RECORD_USR, BASE_RECORD_NAME, BASE_RECORD_PATH,
          BASE_RECORD_TAG_TYPE, BASE_RECORD_IS_VIRTUAL, BASE_RECORD_ACCESS,
          BASE_RECORD_IS_PARENT}},
        // Function Block
        {BI_FUNCTION_BLOCK_ID,
         {FUNCTION_USR, FUNCTION_NAME, FUNCTION_DEFLOCATION, FUNCTION_LOCATION,
          FUNCTION_ACCESS, FUNCTION_IS_METHOD, FUNCTION_IS_STATIC}},
        // Reference Block
        {BI_REFERENCE_BLOCK_ID,
         {REFERENCE_USR, REFERENCE_NAME, REFERENCE_QUAL_NAME, REFERENCE_TYPE,
          REFERENCE_PATH, REFERENCE_FIELD}},
        // Template Blocks.
        {BI_TEMPLATE_BLOCK_ID, {}},
        {BI_TEMPLATE_PARAM_BLOCK_ID, {TEMPLATE_PARAM_CONTENTS}},
        {BI_TEMPLATE_SPECIALIZATION_BLOCK_ID, {TEMPLATE_SPECIALIZATION_OF}}};

// AbbreviationMap

constexpr unsigned char BitCodeConstants::Signature[];

void ClangDocBitcodeWriter::AbbreviationMap::add(RecordId RID,
                                                 unsigned AbbrevID) {
  assert(RecordIdNameMap[RID] && "Unknown RecordId.");
  assert(!Abbrevs.contains(RID) && "Abbreviation already added.");
  Abbrevs[RID] = AbbrevID;
}

unsigned ClangDocBitcodeWriter::AbbreviationMap::get(RecordId RID) const {
  assert(RecordIdNameMap[RID] && "Unknown RecordId.");
  assert(Abbrevs.contains(RID) && "Unknown abbreviation.");
  return Abbrevs.lookup(RID);
}

// Validation and Overview Blocks

/// Emits the magic number header to check that its the right format,
/// in this case, 'DOCS'.
void ClangDocBitcodeWriter::emitHeader() {
  for (char C : BitCodeConstants::Signature)
    Stream.Emit((unsigned)C, BitCodeConstants::SignatureBitSize);
}

void ClangDocBitcodeWriter::emitVersionBlock() {
  StreamSubBlockGuard Block(Stream, BI_VERSION_BLOCK_ID);
  emitRecord(VersionNumber, VERSION);
}

/// Emits a block ID and the block name to the BLOCKINFO block.
void ClangDocBitcodeWriter::emitBlockID(BlockId BID) {
  const auto &BlockIdName = BlockIdNameMap[BID];
  assert(BlockIdName.data() && BlockIdName.size() && "Unknown BlockId.");

  Record.clear();
  Record.push_back(BID);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETBID, Record);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_BLOCKNAME,
                    ArrayRef<unsigned char>(BlockIdName.bytes_begin(),
                                            BlockIdName.bytes_end()));
}

/// Emits a record name to the BLOCKINFO block.
void ClangDocBitcodeWriter::emitRecordID(RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  prepRecordData(ID);
  Record.append(RecordIdNameMap[ID].Name.begin(),
                RecordIdNameMap[ID].Name.end());
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETRECORDNAME, Record);
}

// Abbreviations

void ClangDocBitcodeWriter::emitAbbrev(RecordId ID, BlockId Block) {
  assert(RecordIdNameMap[ID] && "Unknown abbreviation.");
  auto Abbrev = std::make_shared<llvm::BitCodeAbbrev>();
  Abbrev->Add(llvm::BitCodeAbbrevOp(ID));
  RecordIdNameMap[ID].Abbrev(Abbrev);
  Abbrevs.add(ID, Stream.EmitBlockInfoAbbrev(Block, std::move(Abbrev)));
}

// Records

void ClangDocBitcodeWriter::emitRecord(const SymbolID &Sym, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &genSymbolIdAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, Sym != EmptySID))
    return;
  assert(Sym.size() == 20);
  Record.push_back(Sym.size());
  Record.append(Sym.begin(), Sym.end());
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

void ClangDocBitcodeWriter::emitRecord(llvm::StringRef Str, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &genStringAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, !Str.empty()))
    return;
  assert(Str.size() < (1U << BitCodeConstants::StringLengthSize));
  Record.push_back(Str.size());
  Stream.EmitRecordWithBlob(Abbrevs.get(ID), Record, Str);
}

void ClangDocBitcodeWriter::emitRecord(const Location &Loc, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &genLocationAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, true))
    return;
  // FIXME: Assert that the line number is of the appropriate size.
  Record.push_back(Loc.StartLineNumber);
  Record.push_back(Loc.EndLineNumber);
  assert(Loc.Filename.size() < (1U << BitCodeConstants::StringLengthSize));
  Record.push_back(Loc.IsFileInRootDir);
  Record.push_back(Loc.Filename.size());
  Stream.EmitRecordWithBlob(Abbrevs.get(ID), Record, Loc.Filename);
}

void ClangDocBitcodeWriter::emitRecord(bool Val, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &genBoolAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, Val))
    return;
  Record.push_back(Val);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

void ClangDocBitcodeWriter::emitRecord(int Val, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &genIntAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, Val))
    return;
  // FIXME: Assert that the integer is of the appropriate size.
  Record.push_back(Val);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

void ClangDocBitcodeWriter::emitRecord(unsigned Val, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &genIntAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, Val))
    return;
  assert(Val < (1U << BitCodeConstants::IntSize));
  Record.push_back(Val);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

void ClangDocBitcodeWriter::emitRecord(const TemplateInfo &Templ) {}

bool ClangDocBitcodeWriter::prepRecordData(RecordId ID, bool ShouldEmit) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  if (!ShouldEmit)
    return false;
  Record.clear();
  Record.push_back(ID);
  return true;
}

// BlockInfo Block

void ClangDocBitcodeWriter::emitBlockInfoBlock() {
  Stream.EnterBlockInfoBlock();
  for (const auto &Block : RecordsByBlock) {
    assert(Block.second.size() < (1U << BitCodeConstants::SubblockIDSize));
    emitBlockInfo(Block.first, Block.second);
  }
  Stream.ExitBlock();
}

void ClangDocBitcodeWriter::emitBlockInfo(BlockId BID,
                                          const std::vector<RecordId> &RIDs) {
  assert(RIDs.size() < (1U << BitCodeConstants::SubblockIDSize));
  emitBlockID(BID);
  for (RecordId RID : RIDs) {
    emitRecordID(RID);
    emitAbbrev(RID, BID);
  }
}

// Block emission

void ClangDocBitcodeWriter::emitBlock(const Reference &R, FieldId Field) {
  if (R.USR == EmptySID && R.Name.empty())
    return;
  StreamSubBlockGuard Block(Stream, BI_REFERENCE_BLOCK_ID);
  emitRecord(R.USR, REFERENCE_USR);
  emitRecord(R.Name, REFERENCE_NAME);
  emitRecord(R.QualName, REFERENCE_QUAL_NAME);
  emitRecord((unsigned)R.RefType, REFERENCE_TYPE);
  emitRecord(R.Path, REFERENCE_PATH);
  emitRecord((unsigned)Field, REFERENCE_FIELD);
}

void ClangDocBitcodeWriter::emitBlock(const TypeInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_TYPE_BLOCK_ID);
  emitBlock(T.Type, FieldId::F_type);
}

void ClangDocBitcodeWriter::emitBlock(const TypedefInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_TYPEDEF_BLOCK_ID);
  emitRecord(T.USR, TYPEDEF_USR);
  emitRecord(T.Name, TYPEDEF_NAME);
  for (const auto &N : T.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : T.Description)
    emitBlock(CI);
  if (T.DefLoc)
    emitRecord(*T.DefLoc, TYPEDEF_DEFLOCATION);
  emitRecord(T.IsUsing, TYPEDEF_IS_USING);
  emitBlock(T.Underlying);
}

void ClangDocBitcodeWriter::emitBlock(const FieldTypeInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_FIELD_TYPE_BLOCK_ID);
  emitBlock(T.Type, FieldId::F_type);
  emitRecord(T.Name, FIELD_TYPE_NAME);
  emitRecord(T.DefaultValue, FIELD_DEFAULT_VALUE);
}

void ClangDocBitcodeWriter::emitBlock(const MemberTypeInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_MEMBER_TYPE_BLOCK_ID);
  emitBlock(T.Type, FieldId::F_type);
  emitRecord(T.Name, MEMBER_TYPE_NAME);
  emitRecord(T.Access, MEMBER_TYPE_ACCESS);
  emitRecord(T.IsStatic, MEMBER_TYPE_IS_STATIC);
  for (const auto &CI : T.Description)
    emitBlock(CI);
}

void ClangDocBitcodeWriter::emitBlock(const CommentInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_COMMENT_BLOCK_ID);
  // Handle Kind (enum) separately, since it is not a string.
  emitRecord(commentKindToString(I.Kind), COMMENT_KIND);
  for (const auto &L : std::vector<std::pair<llvm::StringRef, RecordId>>{
           {I.Text, COMMENT_TEXT},
           {I.Name, COMMENT_NAME},
           {I.Direction, COMMENT_DIRECTION},
           {I.ParamName, COMMENT_PARAMNAME},
           {I.CloseName, COMMENT_CLOSENAME}})
    emitRecord(L.first, L.second);
  emitRecord(I.SelfClosing, COMMENT_SELFCLOSING);
  emitRecord(I.Explicit, COMMENT_EXPLICIT);
  for (const auto &A : I.AttrKeys)
    emitRecord(A, COMMENT_ATTRKEY);
  for (const auto &A : I.AttrValues)
    emitRecord(A, COMMENT_ATTRVAL);
  for (const auto &A : I.Args)
    emitRecord(A, COMMENT_ARG);
  for (const auto &C : I.Children)
    emitBlock(*C);
}

void ClangDocBitcodeWriter::emitBlock(const NamespaceInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_NAMESPACE_BLOCK_ID);
  emitRecord(I.USR, NAMESPACE_USR);
  emitRecord(I.Name, NAMESPACE_NAME);
  emitRecord(I.Path, NAMESPACE_PATH);
  for (const auto &N : I.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : I.Description)
    emitBlock(CI);
  for (const auto &C : I.Children.Namespaces)
    emitBlock(C, FieldId::F_child_namespace);
  for (const auto &C : I.Children.Records)
    emitBlock(C, FieldId::F_child_record);
  for (const auto &C : I.Children.Functions)
    emitBlock(C);
  for (const auto &C : I.Children.Enums)
    emitBlock(C);
  for (const auto &C : I.Children.Typedefs)
    emitBlock(C);
}

void ClangDocBitcodeWriter::emitBlock(const EnumInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_ENUM_BLOCK_ID);
  emitRecord(I.USR, ENUM_USR);
  emitRecord(I.Name, ENUM_NAME);
  for (const auto &N : I.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : I.Description)
    emitBlock(CI);
  if (I.DefLoc)
    emitRecord(*I.DefLoc, ENUM_DEFLOCATION);
  for (const auto &L : I.Loc)
    emitRecord(L, ENUM_LOCATION);
  emitRecord(I.Scoped, ENUM_SCOPED);
  if (I.BaseType)
    emitBlock(*I.BaseType);
  for (const auto &N : I.Members)
    emitBlock(N);
}

void ClangDocBitcodeWriter::emitBlock(const EnumValueInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_ENUM_VALUE_BLOCK_ID);
  emitRecord(I.Name, ENUM_VALUE_NAME);
  emitRecord(I.Value, ENUM_VALUE_VALUE);
  emitRecord(I.ValueExpr, ENUM_VALUE_EXPR);
  for (const auto &CI : I.Description)
    emitBlock(CI);
}

void ClangDocBitcodeWriter::emitBlock(const RecordInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_RECORD_BLOCK_ID);
  emitRecord(I.USR, RECORD_USR);
  emitRecord(I.Name, RECORD_NAME);
  emitRecord(I.Path, RECORD_PATH);
  for (const auto &N : I.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : I.Description)
    emitBlock(CI);
  if (I.DefLoc)
    emitRecord(*I.DefLoc, RECORD_DEFLOCATION);
  for (const auto &L : I.Loc)
    emitRecord(L, RECORD_LOCATION);
  emitRecord(llvm::to_underlying(I.TagType), RECORD_TAG_TYPE);
  emitRecord(I.IsTypeDef, RECORD_IS_TYPE_DEF);
  for (const auto &N : I.Members)
    emitBlock(N);
  for (const auto &P : I.Parents)
    emitBlock(P, FieldId::F_parent);
  for (const auto &P : I.VirtualParents)
    emitBlock(P, FieldId::F_vparent);
  for (const auto &PB : I.Bases)
    emitBlock(PB);
  for (const auto &C : I.Children.Records)
    emitBlock(C, FieldId::F_child_record);
  for (const auto &C : I.Children.Functions)
    emitBlock(C);
  for (const auto &C : I.Children.Enums)
    emitBlock(C);
  for (const auto &C : I.Children.Typedefs)
    emitBlock(C);
  if (I.Template)
    emitBlock(*I.Template);
}

void ClangDocBitcodeWriter::emitBlock(const BaseRecordInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_BASE_RECORD_BLOCK_ID);
  emitRecord(I.USR, BASE_RECORD_USR);
  emitRecord(I.Name, BASE_RECORD_NAME);
  emitRecord(I.Path, BASE_RECORD_PATH);
  emitRecord(llvm::to_underlying(I.TagType), BASE_RECORD_TAG_TYPE);
  emitRecord(I.IsVirtual, BASE_RECORD_IS_VIRTUAL);
  emitRecord(I.Access, BASE_RECORD_ACCESS);
  emitRecord(I.IsParent, BASE_RECORD_IS_PARENT);
  for (const auto &M : I.Members)
    emitBlock(M);
  for (const auto &C : I.Children.Functions)
    emitBlock(C);
}

void ClangDocBitcodeWriter::emitBlock(const FunctionInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_FUNCTION_BLOCK_ID);
  emitRecord(I.USR, FUNCTION_USR);
  emitRecord(I.Name, FUNCTION_NAME);
  for (const auto &N : I.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : I.Description)
    emitBlock(CI);
  emitRecord(I.Access, FUNCTION_ACCESS);
  emitRecord(I.IsMethod, FUNCTION_IS_METHOD);
  emitRecord(I.IsStatic, FUNCTION_IS_STATIC);
  if (I.DefLoc)
    emitRecord(*I.DefLoc, FUNCTION_DEFLOCATION);
  for (const auto &L : I.Loc)
    emitRecord(L, FUNCTION_LOCATION);
  emitBlock(I.Parent, FieldId::F_parent);
  emitBlock(I.ReturnType);
  for (const auto &N : I.Params)
    emitBlock(N);
  if (I.Template)
    emitBlock(*I.Template);
}

void ClangDocBitcodeWriter::emitBlock(const TemplateInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_TEMPLATE_BLOCK_ID);
  for (const auto &P : T.Params)
    emitBlock(P);
  if (T.Specialization)
    emitBlock(*T.Specialization);
}

void ClangDocBitcodeWriter::emitBlock(const TemplateSpecializationInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_TEMPLATE_SPECIALIZATION_BLOCK_ID);
  emitRecord(T.SpecializationOf, TEMPLATE_SPECIALIZATION_OF);
  for (const auto &P : T.Params)
    emitBlock(P);
}

void ClangDocBitcodeWriter::emitBlock(const TemplateParamInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_TEMPLATE_PARAM_BLOCK_ID);
  emitRecord(T.Contents, TEMPLATE_PARAM_CONTENTS);
}

bool ClangDocBitcodeWriter::dispatchInfoForWrite(Info *I) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    emitBlock(*static_cast<clang::doc::NamespaceInfo *>(I));
    break;
  case InfoType::IT_record:
    emitBlock(*static_cast<clang::doc::RecordInfo *>(I));
    break;
  case InfoType::IT_enum:
    emitBlock(*static_cast<clang::doc::EnumInfo *>(I));
    break;
  case InfoType::IT_function:
    emitBlock(*static_cast<clang::doc::FunctionInfo *>(I));
    break;
  case InfoType::IT_typedef:
    emitBlock(*static_cast<clang::doc::TypedefInfo *>(I));
    break;
  case InfoType::IT_default:
    llvm::errs() << "Unexpected info, unable to write.\n";
    return true;
  }
  return false;
}

} // namespace doc
} // namespace clang
