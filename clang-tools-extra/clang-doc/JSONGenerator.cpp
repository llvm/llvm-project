#include "Generators.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::json;

namespace clang {
namespace doc {

class JSONGenerator : public Generator {
public:
  static const char *Format;

  Error generateDocs(StringRef RootDir,
                     llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                     const ClangDocContext &CDCtx) override;
  Error createResources(ClangDocContext &CDCtx) override;
  Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                           const ClangDocContext &CDCtx) override;
};

const char *JSONGenerator::Format = "json";

static void serializeInfo(const TypedefInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl);
static void serializeInfo(const EnumInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl);

static json::Object serializeLocation(const Location &Loc,
                                      std::optional<StringRef> RepositoryUrl) {
  Object LocationObj = Object();
  LocationObj["LineNumber"] = Loc.StartLineNumber;
  LocationObj["Filename"] = Loc.Filename;

  if (!Loc.IsFileInRootDir || !RepositoryUrl)
    return LocationObj;
  SmallString<128> FileURL(*RepositoryUrl);
  sys::path::append(FileURL, sys::path::Style::posix, Loc.Filename);
  FileURL += "#" + std::to_string(Loc.StartLineNumber);
  LocationObj["FileURL"] = FileURL;
  return LocationObj;
}

static json::Value serializeComment(const CommentInfo &Comment) {
  assert((Comment.Kind == "BlockCommandComment" ||
          Comment.Kind == "FullComment" || Comment.Kind == "ParagraphComment" ||
          Comment.Kind == "TextComment") &&
         "Unknown Comment type in CommentInfo.");

  Object Obj = Object();
  json::Value Child = Object();

  // TextComment has no children, so return it.
  if (Comment.Kind == "TextComment") {
    Obj["TextComment"] = Comment.Text;
    return Obj;
  }

  // BlockCommandComment needs to generate a Command key.
  if (Comment.Kind == "BlockCommandComment")
    Child.getAsObject()->insert({"Command", Comment.Name});

  // Use the same handling for everything else.
  // Only valid for:
  //  - BlockCommandComment
  //  - FullComment
  //  - ParagraphComment
  json::Value ChildArr = Array();
  auto &CARef = *ChildArr.getAsArray();
  CARef.reserve(Comment.Children.size());
  for (const auto &C : Comment.Children)
    CARef.emplace_back(serializeComment(*C));
  Child.getAsObject()->insert({"Children", ChildArr});
  Obj.insert({Comment.Kind, Child});
  return Obj;
}

static void serializeCommonAttributes(const Info &I, json::Object &Obj,
                                      std::optional<StringRef> RepositoryUrl) {
  Obj["Name"] = I.Name.str();
  Obj["USR"] = toHex(toStringRef(I.USR));

  if (!I.Path.empty())
    Obj["Path"] = I.Path.str();

  if (!I.Namespace.empty()) {
    Obj["Namespace"] = json::Array();
    for (const auto &NS : I.Namespace)
      Obj["Namespace"].getAsArray()->push_back(NS.Name.str());
  }

  if (!I.Description.empty()) {
    json::Value DescArray = json::Array();
    auto &DescArrayRef = *DescArray.getAsArray();
    for (const auto &Comment : I.Description)
      DescArrayRef.push_back(serializeComment(Comment));
    Obj["Description"] = std::move(DescArray);
  }

  // Namespaces aren't SymbolInfos, so they dont have a DefLoc
  if (I.IT != InfoType::IT_namespace) {
    const auto *Symbol = static_cast<const SymbolInfo *>(&I);
    if (Symbol->DefLoc)
      Obj["Location"] =
          serializeLocation(Symbol->DefLoc.value(), RepositoryUrl);
  }
}

static void serializeReference(const Reference &Ref, Object &ReferenceObj,
                               SmallString<64> CurrentDirectory) {
  SmallString<64> Path = Ref.getRelativeFilePath(CurrentDirectory);
  sys::path::append(Path, Ref.getFileBaseName() + ".json");
  sys::path::native(Path, sys::path::Style::posix);
  ReferenceObj["Link"] = Path;
  ReferenceObj["Name"] = Ref.Name;
  ReferenceObj["QualName"] = Ref.QualName;
  ReferenceObj["ID"] = toHex(toStringRef(Ref.USR));
}

static void serializeReference(Object &Obj, SmallVector<Reference, 4> References, std::string Key) {
  json::Value ReferencesArray = Array();
  json::Array &ReferencesArrayRef = *ReferencesArray.getAsArray();
  for (const auto& Reference : References) {
    Object ReferenceObject = Object();
    auto BasePath = Reference.getRelativeFilePath("");
    serializeReference(Reference, ReferenceObject, BasePath);
    ReferencesArrayRef.push_back(std::move(ReferenceObject));
  } 
  Obj[Key] = std::move(ReferencesArray);
}

// Although namespaces and records both have ScopeChildren, they serialize them
// differently. Only enums, records, and typedefs are handled here.
static void serializeCommonChildren(const ScopeChildren &Children,
                                    json::Object &Obj,
                                    std::optional<StringRef> RepositoryUrl) {
  if (!Children.Enums.empty()) {
    json::Value EnumsArray = Array();
    auto &EnumsArrayRef = *EnumsArray.getAsArray();
    for (const auto &Enum : Children.Enums) {
      json::Object EnumObj;
      serializeInfo(Enum, EnumObj, RepositoryUrl);
      EnumsArrayRef.push_back(std::move(EnumObj));
    }
    Obj["Enums"] = std::move(EnumsArray);
  }

  if (!Children.Typedefs.empty()) {
    json::Value TypedefsArray = Array();
    auto &TypedefsArrayRef = *TypedefsArray.getAsArray();
    for (const auto &Typedef : Children.Typedefs) {
      json::Object TypedefObj;
      serializeInfo(Typedef, TypedefObj, RepositoryUrl);
      TypedefsArrayRef.push_back(std::move(TypedefObj));
    }
    Obj["Typedefs"] = std::move(TypedefsArray);
  }

  if (!Children.Records.empty()) {
    json::Value RecordsArray = Array();
    auto &RecordsArrayRef = *RecordsArray.getAsArray();
    for (const auto &Record : Children.Records) {
      json::Object RecordObj;
      SmallString<64> BasePath = Record.getRelativeFilePath("");
      serializeReference(Record, RecordObj, BasePath);
      RecordsArrayRef.push_back(std::move(RecordObj));
    }
    Obj["Records"] = std::move(RecordsArray);
  }
}

static void serializeInfo(const TypeInfo &I, Object &Obj) {
  Obj["Name"] = I.Type.Name;
  Obj["QualName"] = I.Type.QualName;
  Obj["ID"] = toHex(toStringRef(I.Type.USR));
  Obj["IsTemplate"] = I.IsTemplate;
  Obj["IsBuiltIn"] = I.IsBuiltIn;
}

static void serializeInfo(const FunctionInfo &F, json::Object &Obj,
                          std::optional<StringRef> RepositoryURL) {
  serializeCommonAttributes(F, Obj, RepositoryURL);
  Obj["IsStatic"] = F.IsStatic;

  auto ReturnTypeObj = Object();
  serializeInfo(F.ReturnType, ReturnTypeObj);
  Obj["ReturnType"] = std::move(ReturnTypeObj);

  if (!F.Params.empty()) {
    json::Value ParamsArray = json::Array();
    auto &ParamsArrayRef = *ParamsArray.getAsArray();
    for (const auto &Param : F.Params) {
      json::Object ParamObj;
      ParamObj["Name"] = Param.Name;
      ParamObj["Type"] = Param.Type.Name;
      ParamsArrayRef.push_back(std::move(ParamObj));
    }
    Obj["Params"] = std::move(ParamsArray);
  }
}

static void serializeInfo(const EnumInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["Scoped"] = I.Scoped;

  if (I.BaseType) {
    json::Object BaseTypeObj;
    BaseTypeObj["Name"] = I.BaseType->Type.Name;
    BaseTypeObj["QualName"] = I.BaseType->Type.QualName;
    BaseTypeObj["ID"] = toHex(toStringRef(I.BaseType->Type.USR));
    Obj["BaseType"] = std::move(BaseTypeObj);
  }

  if (!I.Members.empty()) {
    json::Value MembersArray = Array();
    auto &MembersArrayRef = *MembersArray.getAsArray();
    for (const auto &Member : I.Members) {
      json::Object MemberObj;
      MemberObj["Name"] = Member.Name;
      if (!Member.ValueExpr.empty())
        MemberObj["ValueExpr"] = Member.ValueExpr;
      else
        MemberObj["Value"] = Member.Value;
      MembersArrayRef.push_back(std::move(MemberObj));
    }
    Obj["Members"] = std::move(MembersArray);
  }
}

static void serializeInfo(const TypedefInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["TypeDeclaration"] = I.TypeDeclaration;
  Obj["IsUsing"] = I.IsUsing;
  Object TypeObj = Object();
  serializeInfo(I.Underlying, TypeObj);
  Obj["Underlying"] = std::move(TypeObj);
}

static void serializeInfo(const RecordInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["FullName"] = I.Name.str();
  Obj["TagType"] = getTagType(I.TagType);
  Obj["IsTypedef"] = I.IsTypeDef;

  if (!I.Children.Functions.empty()) {
    json::Value PublicFunctionArr = Array();
    json::Array &PublicFunctionARef = *PublicFunctionArr.getAsArray();
    json::Value ProtectedFunctionArr = Array();
    json::Array &ProtectedFunctionARef = *ProtectedFunctionArr.getAsArray();

    for (const auto &Function : I.Children.Functions) {
      json::Object FunctionObj;
      serializeInfo(Function, FunctionObj, RepositoryUrl);
      AccessSpecifier Access = Function.Access;
      if (Access == AccessSpecifier::AS_public)
        PublicFunctionARef.push_back(std::move(FunctionObj));
      else if (Access == AccessSpecifier::AS_protected)
        ProtectedFunctionARef.push_back(std::move(FunctionObj));
    }

    if (!PublicFunctionARef.empty())
      Obj["PublicFunctions"] = std::move(PublicFunctionArr);
    if (!ProtectedFunctionARef.empty())
      Obj["ProtectedFunctions"] = std::move(ProtectedFunctionArr);
  }

  if (!I.Members.empty()) {
    json::Value PublicMembers = Array();
    json::Array &PubMemberRef = *PublicMembers.getAsArray();
    json::Value ProtectedMembers = Array();
    json::Array &ProtMemberRef = *ProtectedMembers.getAsArray();

    for (const MemberTypeInfo &Member : I.Members) {
      json::Object MemberObj = Object();
      MemberObj["Name"] = Member.Name;
      MemberObj["Type"] = Member.Type.Name;

      if (Member.Access == AccessSpecifier::AS_public)
        PubMemberRef.push_back(std::move(MemberObj));
      else if (Member.Access == AccessSpecifier::AS_protected)
        ProtMemberRef.push_back(std::move(MemberObj));
    }

    if (!PubMemberRef.empty())
      Obj["PublicMembers"] = std::move(PublicMembers);
    if (!ProtMemberRef.empty())
      Obj["ProtectedMembers"] = std::move(ProtectedMembers);
  }

  if (!I.Bases.empty()) {
    json::Value BasesArray = Array();
    json::Array &BasesArrayRef = *BasesArray.getAsArray();
    for (const auto &BaseInfo : I.Bases) {
      Object BaseInfoObj = Object();
      serializeInfo(BaseInfo, BaseInfoObj, RepositoryUrl);
      BaseInfoObj["IsVirtual"] = BaseInfo.IsVirtual;
      BaseInfoObj["Access"] = getAccessSpelling(BaseInfo.Access);
      BaseInfoObj["IsParent"] = BaseInfo.IsParent;
      BasesArrayRef.push_back(std::move(BaseInfoObj));
    }
    Obj["Bases"] = std::move(BasesArray);
  }

  if (!I.Parents.empty())
    serializeReference(Obj, I.Parents, "Parents");

  if (!I.VirtualParents.empty())
    serializeReference(Obj, I.VirtualParents, "VirtualParents");

  serializeCommonChildren(I.Children, Obj, RepositoryUrl);
}

Error JSONGenerator::generateDocs(
    StringRef RootDir, llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
    const ClangDocContext &CDCtx) {
  StringSet<> CreatedDirs;
  StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    Info *Info = Group.getValue().get();

    SmallString<128> Path;
    sys::path::native(RootDir, Path);
    sys::path::append(Path, Info->getRelativeFilePath(""));
    if (!CreatedDirs.contains(Path)) {
      if (std::error_code Err = sys::fs::create_directories(Path);
          Err != std::error_code())
        return createFileError(Twine(Path), Err);
      CreatedDirs.insert(Path);
    }

    sys::path::append(Path, Info->getFileBaseName() + ".json");
    FileToInfos[Path].push_back(Info);
  }

  for (const auto &Group : FileToInfos) {
    std::error_code FileErr;
    raw_fd_ostream InfoOS(Group.getKey(), FileErr, sys::fs::OF_Text);
    if (FileErr)
      return createFileError("cannot open file " + Group.getKey(), FileErr);

    for (const auto &Info : Group.getValue())
      if (Error Err = generateDocForInfo(Info, InfoOS, CDCtx))
        return Err;
  }

  return Error::success();
}

Error JSONGenerator::generateDocForInfo(Info *I, raw_ostream &OS,
                                        const ClangDocContext &CDCtx) {
  json::Object Obj = Object();

  switch (I->IT) {
  case InfoType::IT_namespace:
    break;
  case InfoType::IT_record:
    serializeInfo(*static_cast<RecordInfo *>(I), Obj, CDCtx.RepositoryUrl);
    break;
  case InfoType::IT_enum:
  case InfoType::IT_function:
  case InfoType::IT_typedef:
    break;
  case InfoType::IT_default:
    return createStringError(inconvertibleErrorCode(), "unexpected info type");
  }
  OS << llvm::formatv("{0:2}", llvm::json::Value(std::move(Obj)));
  return Error::success();
}

Error JSONGenerator::createResources(ClangDocContext &CDCtx) {
  return Error::success();
}

static GeneratorRegistry::Add<JSONGenerator> JSON(JSONGenerator::Format,
                                                  "Generator for JSON output.");
volatile int JSONGeneratorAnchorSource = 0;
} // namespace doc
} // namespace clang
