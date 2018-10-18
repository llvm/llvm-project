//===--  BitcodeWriter.cpp - ClangDoc Bitcode Writer ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

static void AbbrevGen(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev,
                      const std::initializer_list<llvm::BitCodeAbbrevOp> Ops) {
  for (const auto &Op : Ops)
    Abbrev->Add(Op);
}

static void BoolAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  AbbrevGen(Abbrev,
            {// 0. Boolean
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                   BitCodeConstants::BoolSize)});
}

static void IntAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  AbbrevGen(Abbrev,
            {// 0. Fixed-size integer
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                   BitCodeConstants::IntSize)});
}

static void SymbolIDAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  AbbrevGen(Abbrev,
            {// 0. Fixed-size integer (length of the sha1'd USR)
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                   BitCodeConstants::USRLengthSize),
             // 1. Fixed-size array of Char6 (USR)
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Array),
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                   BitCodeConstants::USRBitLengthSize)});
}

static void StringAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  AbbrevGen(Abbrev,
            {// 0. Fixed-size integer (length of the following string)
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                                   BitCodeConstants::StringLengthSize),
             // 1. The string blob
             llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Blob)});
}

// Assumes that the file will not have more than 65535 lines.
static void LocationAbbrev(std::shared_ptr<llvm::BitCodeAbbrev> &Abbrev) {
  AbbrevGen(
      Abbrev,
      {// 0. Fixed-size integer (line number)
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                             BitCodeConstants::LineNumberSize),
       // 1. Fixed-size integer (length of the following string (filename))
       llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed,
                             BitCodeConstants::StringLengthSize),
       // 2. The string blob
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
          {BI_TYPE_BLOCK_ID, "TypeBlock"},
          {BI_FIELD_TYPE_BLOCK_ID, "FieldTypeBlock"},
          {BI_MEMBER_TYPE_BLOCK_ID, "MemberTypeBlock"},
          {BI_RECORD_BLOCK_ID, "RecordBlock"},
          {BI_FUNCTION_BLOCK_ID, "FunctionBlock"},
          {BI_COMMENT_BLOCK_ID, "CommentBlock"},
          {BI_REFERENCE_BLOCK_ID, "ReferenceBlock"}};
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
          {VERSION, {"Version", &IntAbbrev}},
          {COMMENT_KIND, {"Kind", &StringAbbrev}},
          {COMMENT_TEXT, {"Text", &StringAbbrev}},
          {COMMENT_NAME, {"Name", &StringAbbrev}},
          {COMMENT_DIRECTION, {"Direction", &StringAbbrev}},
          {COMMENT_PARAMNAME, {"ParamName", &StringAbbrev}},
          {COMMENT_CLOSENAME, {"CloseName", &StringAbbrev}},
          {COMMENT_SELFCLOSING, {"SelfClosing", &BoolAbbrev}},
          {COMMENT_EXPLICIT, {"Explicit", &BoolAbbrev}},
          {COMMENT_ATTRKEY, {"AttrKey", &StringAbbrev}},
          {COMMENT_ATTRVAL, {"AttrVal", &StringAbbrev}},
          {COMMENT_ARG, {"Arg", &StringAbbrev}},
          {FIELD_TYPE_NAME, {"Name", &StringAbbrev}},
          {MEMBER_TYPE_NAME, {"Name", &StringAbbrev}},
          {MEMBER_TYPE_ACCESS, {"Access", &IntAbbrev}},
          {NAMESPACE_USR, {"USR", &SymbolIDAbbrev}},
          {NAMESPACE_NAME, {"Name", &StringAbbrev}},
          {ENUM_USR, {"USR", &SymbolIDAbbrev}},
          {ENUM_NAME, {"Name", &StringAbbrev}},
          {ENUM_DEFLOCATION, {"DefLocation", &LocationAbbrev}},
          {ENUM_LOCATION, {"Location", &LocationAbbrev}},
          {ENUM_MEMBER, {"Member", &StringAbbrev}},
          {ENUM_SCOPED, {"Scoped", &BoolAbbrev}},
          {RECORD_USR, {"USR", &SymbolIDAbbrev}},
          {RECORD_NAME, {"Name", &StringAbbrev}},
          {RECORD_DEFLOCATION, {"DefLocation", &LocationAbbrev}},
          {RECORD_LOCATION, {"Location", &LocationAbbrev}},
          {RECORD_TAG_TYPE, {"TagType", &IntAbbrev}},
          {FUNCTION_USR, {"USR", &SymbolIDAbbrev}},
          {FUNCTION_NAME, {"Name", &StringAbbrev}},
          {FUNCTION_DEFLOCATION, {"DefLocation", &LocationAbbrev}},
          {FUNCTION_LOCATION, {"Location", &LocationAbbrev}},
          {FUNCTION_ACCESS, {"Access", &IntAbbrev}},
          {FUNCTION_IS_METHOD, {"IsMethod", &BoolAbbrev}},
          {REFERENCE_USR, {"USR", &SymbolIDAbbrev}},
          {REFERENCE_NAME, {"Name", &StringAbbrev}},
          {REFERENCE_TYPE, {"RefType", &IntAbbrev}},
          {REFERENCE_FIELD, {"Field", &IntAbbrev}}};
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
        {BI_FIELD_TYPE_BLOCK_ID, {FIELD_TYPE_NAME}},
        // MemberType Block
        {BI_MEMBER_TYPE_BLOCK_ID, {MEMBER_TYPE_NAME, MEMBER_TYPE_ACCESS}},
        // Enum Block
        {BI_ENUM_BLOCK_ID,
         {ENUM_USR, ENUM_NAME, ENUM_DEFLOCATION, ENUM_LOCATION, ENUM_MEMBER,
          ENUM_SCOPED}},
        // Namespace Block
        {BI_NAMESPACE_BLOCK_ID, {NAMESPACE_USR, NAMESPACE_NAME}},
        // Record Block
        {BI_RECORD_BLOCK_ID,
         {RECORD_USR, RECORD_NAME, RECORD_DEFLOCATION, RECORD_LOCATION,
          RECORD_TAG_TYPE}},
        // Function Block
        {BI_FUNCTION_BLOCK_ID,
         {FUNCTION_USR, FUNCTION_NAME, FUNCTION_DEFLOCATION, FUNCTION_LOCATION,
          FUNCTION_ACCESS, FUNCTION_IS_METHOD}},
        // Reference Block
        {BI_REFERENCE_BLOCK_ID,
         {REFERENCE_USR, REFERENCE_NAME, REFERENCE_TYPE, REFERENCE_FIELD}}};

// AbbreviationMap

constexpr char BitCodeConstants::Signature[];

void ClangDocBitcodeWriter::AbbreviationMap::add(RecordId RID,
                                                 unsigned AbbrevID) {
  assert(RecordIdNameMap[RID] && "Unknown RecordId.");
  assert(Abbrevs.find(RID) == Abbrevs.end() && "Abbreviation already added.");
  Abbrevs[RID] = AbbrevID;
}

unsigned ClangDocBitcodeWriter::AbbreviationMap::get(RecordId RID) const {
  assert(RecordIdNameMap[RID] && "Unknown RecordId.");
  assert(Abbrevs.find(RID) != Abbrevs.end() && "Unknown abbreviation.");
  return Abbrevs.lookup(RID);
}

// Validation and Overview Blocks

/// \brief Emits the magic number header to check that its the right format,
/// in this case, 'DOCS'.
void ClangDocBitcodeWriter::emitHeader() {
  for (char C : BitCodeConstants::Signature)
    Stream.Emit((unsigned)C, BitCodeConstants::SignatureBitSize);
}

void ClangDocBitcodeWriter::emitVersionBlock() {
  StreamSubBlockGuard Block(Stream, BI_VERSION_BLOCK_ID);
  emitRecord(VersionNumber, VERSION);
}

/// \brief Emits a block ID and the block name to the BLOCKINFO block.
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

/// \brief Emits a record name to the BLOCKINFO block.
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
  assert(RecordIdNameMap[ID].Abbrev == &SymbolIDAbbrev &&
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
  assert(RecordIdNameMap[ID].Abbrev == &StringAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, !Str.empty()))
    return;
  assert(Str.size() < (1U << BitCodeConstants::StringLengthSize));
  Record.push_back(Str.size());
  Stream.EmitRecordWithBlob(Abbrevs.get(ID), Record, Str);
}

void ClangDocBitcodeWriter::emitRecord(const Location &Loc, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &LocationAbbrev &&
         "Abbrev type mismatch.");
  if (!prepRecordData(ID, true))
    return;
  // FIXME: Assert that the line number is of the appropriate size.
  Record.push_back(Loc.LineNumber);
  assert(Loc.Filename.size() < (1U << BitCodeConstants::StringLengthSize));
  Record.push_back(Loc.Filename.size());
  Stream.EmitRecordWithBlob(Abbrevs.get(ID), Record, Loc.Filename);
}

void ClangDocBitcodeWriter::emitRecord(bool Val, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &BoolAbbrev && "Abbrev type mismatch.");
  if (!prepRecordData(ID, Val))
    return;
  Record.push_back(Val);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

void ClangDocBitcodeWriter::emitRecord(int Val, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &IntAbbrev && "Abbrev type mismatch.");
  if (!prepRecordData(ID, Val))
    return;
  // FIXME: Assert that the integer is of the appropriate size.
  Record.push_back(Val);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

void ClangDocBitcodeWriter::emitRecord(unsigned Val, RecordId ID) {
  assert(RecordIdNameMap[ID] && "Unknown RecordId.");
  assert(RecordIdNameMap[ID].Abbrev == &IntAbbrev && "Abbrev type mismatch.");
  if (!prepRecordData(ID, Val))
    return;
  assert(Val < (1U << BitCodeConstants::IntSize));
  Record.push_back(Val);
  Stream.EmitRecordWithAbbrev(Abbrevs.get(ID), Record);
}

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
  emitRecord((unsigned)R.RefType, REFERENCE_TYPE);
  emitRecord((unsigned)Field, REFERENCE_FIELD);
}

void ClangDocBitcodeWriter::emitBlock(const TypeInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_TYPE_BLOCK_ID);
  emitBlock(T.Type, FieldId::F_type);
}

void ClangDocBitcodeWriter::emitBlock(const FieldTypeInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_FIELD_TYPE_BLOCK_ID);
  emitBlock(T.Type, FieldId::F_type);
  emitRecord(T.Name, FIELD_TYPE_NAME);
}

void ClangDocBitcodeWriter::emitBlock(const MemberTypeInfo &T) {
  StreamSubBlockGuard Block(Stream, BI_MEMBER_TYPE_BLOCK_ID);
  emitBlock(T.Type, FieldId::F_type);
  emitRecord(T.Name, MEMBER_TYPE_NAME);
  emitRecord(T.Access, MEMBER_TYPE_ACCESS);
}

void ClangDocBitcodeWriter::emitBlock(const CommentInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_COMMENT_BLOCK_ID);
  for (const auto &L : std::vector<std::pair<llvm::StringRef, RecordId>>{
           {I.Kind, COMMENT_KIND},
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
  for (const auto &N : I.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : I.Description)
    emitBlock(CI);
  for (const auto &C : I.ChildNamespaces)
    emitBlock(C, FieldId::F_child_namespace);
  for (const auto &C : I.ChildRecords)
    emitBlock(C, FieldId::F_child_record);
  for (const auto &C : I.ChildFunctions)
    emitBlock(C);
  for (const auto &C : I.ChildEnums)
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
    emitRecord(I.DefLoc.getValue(), ENUM_DEFLOCATION);
  for (const auto &L : I.Loc)
    emitRecord(L, ENUM_LOCATION);
  emitRecord(I.Scoped, ENUM_SCOPED);
  for (const auto &N : I.Members)
    emitRecord(N, ENUM_MEMBER);
}

void ClangDocBitcodeWriter::emitBlock(const RecordInfo &I) {
  StreamSubBlockGuard Block(Stream, BI_RECORD_BLOCK_ID);
  emitRecord(I.USR, RECORD_USR);
  emitRecord(I.Name, RECORD_NAME);
  for (const auto &N : I.Namespace)
    emitBlock(N, FieldId::F_namespace);
  for (const auto &CI : I.Description)
    emitBlock(CI);
  if (I.DefLoc)
    emitRecord(I.DefLoc.getValue(), RECORD_DEFLOCATION);
  for (const auto &L : I.Loc)
    emitRecord(L, RECORD_LOCATION);
  emitRecord(I.TagType, RECORD_TAG_TYPE);
  for (const auto &N : I.Members)
    emitBlock(N);
  for (const auto &P : I.Parents)
    emitBlock(P, FieldId::F_parent);
  for (const auto &P : I.VirtualParents)
    emitBlock(P, FieldId::F_vparent);
  for (const auto &C : I.ChildRecords)
    emitBlock(C, FieldId::F_child_record);
  for (const auto &C : I.ChildFunctions)
    emitBlock(C);
  for (const auto &C : I.ChildEnums)
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
  emitRecord(I.IsMethod, FUNCTION_IS_METHOD);
  if (I.DefLoc)
    emitRecord(I.DefLoc.getValue(), FUNCTION_DEFLOCATION);
  for (const auto &L : I.Loc)
    emitRecord(L, FUNCTION_LOCATION);
  emitBlock(I.Parent, FieldId::F_parent);
  emitBlock(I.ReturnType);
  for (const auto &N : I.Params)
    emitBlock(N);
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
  default:
    llvm::errs() << "Unexpected info, unable to write.\n";
    return true;
  }
  return false;
}

} // namespace doc
} // namespace clang
