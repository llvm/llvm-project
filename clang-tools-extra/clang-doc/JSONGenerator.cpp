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
  Obj["Name"] = I.Name;
  Obj["USR"] = toHex(toStringRef(I.USR));

  if (!I.Path.empty())
    Obj["Path"] = I.Path;

  if (!I.Namespace.empty()) {
    Obj["Namespace"] = json::Array();
    for (const auto &NS : I.Namespace)
      Obj["Namespace"].getAsArray()->push_back(NS.Name);
  }

  if (!I.Description.empty()) {
    json::Value DescArray = json::Array();
    auto &DescArrayRef = *DescArray.getAsArray();
    DescArrayRef.reserve(I.Description.size());
    for (const auto &Comment : I.Description)
      DescArrayRef.push_back(serializeComment(Comment));
    Obj["Description"] = DescArray;
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
  ReferenceObj["USR"] = toHex(toStringRef(Ref.USR));
}

static void serializeReference(const SmallVector<Reference, 4> &References,
                               Object &Obj, std::string Key) {
  json::Value ReferencesArray = Array();
  json::Array &ReferencesArrayRef = *ReferencesArray.getAsArray();
  ReferencesArrayRef.reserve(References.size());
  for (const auto &Reference : References) {
    json::Value ReferenceVal = Object();
    auto &ReferenceObj = *ReferenceVal.getAsObject();
    auto BasePath = Reference.getRelativeFilePath("");
    serializeReference(Reference, ReferenceObj, BasePath);
    ReferencesArrayRef.push_back(ReferenceVal);
  }
  Obj[Key] = ReferencesArray;
}

// Although namespaces and records both have ScopeChildren, they serialize them
// differently. Only enums, records, and typedefs are handled here.
static void serializeCommonChildren(const ScopeChildren &Children,
                                    json::Object &Obj,
                                    std::optional<StringRef> RepositoryUrl) {
  if (!Children.Enums.empty()) {
    json::Value EnumsArray = Array();
    auto &EnumsArrayRef = *EnumsArray.getAsArray();
    EnumsArrayRef.reserve(Children.Enums.size());
    for (const auto &Enum : Children.Enums) {
      json::Value EnumVal = Object();
      auto &EnumObj = *EnumVal.getAsObject();
      serializeInfo(Enum, EnumObj, RepositoryUrl);
      EnumsArrayRef.push_back(EnumVal);
    }
    Obj["Enums"] = EnumsArray;
  }

  if (!Children.Typedefs.empty()) {
    json::Value TypedefsArray = Array();
    auto &TypedefsArrayRef = *TypedefsArray.getAsArray();
    TypedefsArrayRef.reserve(Children.Typedefs.size());
    for (const auto &Typedef : Children.Typedefs) {
      json::Value TypedefVal = Object();
      auto &TypedefObj = *TypedefVal.getAsObject();
      serializeInfo(Typedef, TypedefObj, RepositoryUrl);
      TypedefsArrayRef.push_back(TypedefVal);
    }
    Obj["Typedefs"] = TypedefsArray;
  }

  if (!Children.Records.empty()) {
    json::Value RecordsArray = Array();
    auto &RecordsArrayRef = *RecordsArray.getAsArray();
    RecordsArrayRef.reserve(Children.Records.size());
    for (const auto &Record : Children.Records) {
      json::Value RecordVal = Object();
      auto &RecordObj = *RecordVal.getAsObject();
      SmallString<64> BasePath = Record.getRelativeFilePath("");
      serializeReference(Record, RecordObj, BasePath);
      RecordsArrayRef.push_back(RecordVal);
    }
    Obj["Records"] = RecordsArray;
  }
}

static void serializeInfo(const TemplateInfo &Template, Object &Obj) {
  json::Value TemplateVal = Object();
  auto &TemplateObj = *TemplateVal.getAsObject();

  if (Template.Specialization) {
    json::Value TemplateSpecializationVal = Object();
    auto &TemplateSpecializationObj = *TemplateSpecializationVal.getAsObject();
    TemplateSpecializationObj["SpecializationOf"] =
        toHex(toStringRef(Template.Specialization->SpecializationOf));
    if (!Template.Specialization->Params.empty()) {
      json::Value ParamsArray = Array();
      auto &ParamsArrayRef = *ParamsArray.getAsArray();
      ParamsArrayRef.reserve(Template.Specialization->Params.size());
      for (const auto &Param : Template.Specialization->Params)
        ParamsArrayRef.push_back(Param.Contents);
      TemplateSpecializationObj["Parameters"] = ParamsArray;
    }
    TemplateObj["Specialization"] = TemplateSpecializationVal;
  }

  if (!Template.Params.empty()) {
    json::Value ParamsArray = Array();
    auto &ParamsArrayRef = *ParamsArray.getAsArray();
    ParamsArrayRef.reserve(Template.Params.size());
    for (const auto &Param : Template.Params)
      ParamsArrayRef.push_back(Param.Contents);
    TemplateObj["Parameters"] = ParamsArray;
  }

  Obj["Template"] = TemplateVal;
}

static void serializeInfo(const TypeInfo &I, Object &Obj) {
  Obj["Name"] = I.Type.Name;
  Obj["QualName"] = I.Type.QualName;
  Obj["USR"] = toHex(toStringRef(I.Type.USR));
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
    ParamsArrayRef.reserve(F.Params.size());
    for (const auto &Param : F.Params) {
      json::Value ParamVal = Object();
      auto &ParamObj = *ParamVal.getAsObject();
      ParamObj["Name"] = Param.Name;
      ParamObj["Type"] = Param.Type.Name;
      ParamsArrayRef.push_back(ParamVal);
    }
    Obj["Params"] = ParamsArray;
  }

  if (F.Template)
    serializeInfo(F.Template.value(), Obj);
}

static void serializeInfo(const EnumInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["Scoped"] = I.Scoped;

  if (I.BaseType) {
    json::Value BaseTypeVal = Object();
    auto &BaseTypeObj = *BaseTypeVal.getAsObject();
    BaseTypeObj["Name"] = I.BaseType->Type.Name;
    BaseTypeObj["QualName"] = I.BaseType->Type.QualName;
    BaseTypeObj["USR"] = toHex(toStringRef(I.BaseType->Type.USR));
    Obj["BaseType"] = BaseTypeVal;
  }

  if (!I.Members.empty()) {
    json::Value MembersArray = Array();
    auto &MembersArrayRef = *MembersArray.getAsArray();
    MembersArrayRef.reserve(I.Members.size());
    for (const auto &Member : I.Members) {
      json::Value MemberVal = Object();
      auto &MemberObj = *MemberVal.getAsObject();
      MemberObj["Name"] = Member.Name;
      if (!Member.ValueExpr.empty())
        MemberObj["ValueExpr"] = Member.ValueExpr;
      else
        MemberObj["Value"] = Member.Value;
      MembersArrayRef.push_back(MemberVal);
    }
    Obj["Members"] = MembersArray;
  }
}

static void serializeInfo(const TypedefInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["TypeDeclaration"] = I.TypeDeclaration;
  Obj["IsUsing"] = I.IsUsing;
  json::Value TypeVal = Object();
  auto &TypeObj = *TypeVal.getAsObject();
  serializeInfo(I.Underlying, TypeObj);
  Obj["Underlying"] = TypeVal;
}

static void serializeInfo(const RecordInfo &I, json::Object &Obj,
                          std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["FullName"] = I.FullName;
  Obj["TagType"] = getTagType(I.TagType);
  Obj["IsTypedef"] = I.IsTypeDef;

  if (!I.Children.Functions.empty()) {
    json::Value PubFunctionsArray = Array();
    json::Array &PubFunctionsArrayRef = *PubFunctionsArray.getAsArray();
    json::Value ProtFunctionsArray = Array();
    json::Array &ProtFunctionsArrayRef = *ProtFunctionsArray.getAsArray();

    for (const auto &Function : I.Children.Functions) {
      json::Value FunctionVal = Object();
      auto &FunctionObj = *FunctionVal.getAsObject();
      serializeInfo(Function, FunctionObj, RepositoryUrl);
      AccessSpecifier Access = Function.Access;
      if (Access == AccessSpecifier::AS_public)
        PubFunctionsArrayRef.push_back(FunctionVal);
      else if (Access == AccessSpecifier::AS_protected)
        ProtFunctionsArrayRef.push_back(FunctionVal);
    }

    if (!PubFunctionsArrayRef.empty())
      Obj["PublicFunctions"] = PubFunctionsArray;
    if (!ProtFunctionsArrayRef.empty())
      Obj["ProtectedFunctions"] = ProtFunctionsArray;
  }

  if (!I.Members.empty()) {
    json::Value PublicMembersArray = Array();
    json::Array &PubMembersArrayRef = *PublicMembersArray.getAsArray();
    json::Value ProtectedMembersArray = Array();
    json::Array &ProtMembersArrayRef = *ProtectedMembersArray.getAsArray();

    for (const MemberTypeInfo &Member : I.Members) {
      json::Value MemberVal = Object();
      auto &MemberObj = *MemberVal.getAsObject();
      MemberObj["Name"] = Member.Name;
      MemberObj["Type"] = Member.Type.Name;

      if (Member.Access == AccessSpecifier::AS_public)
        PubMembersArrayRef.push_back(MemberVal);
      else if (Member.Access == AccessSpecifier::AS_protected)
        ProtMembersArrayRef.push_back(MemberVal);
    }

    if (!PubMembersArrayRef.empty())
      Obj["PublicMembers"] = PublicMembersArray;
    if (!ProtMembersArrayRef.empty())
      Obj["ProtectedMembers"] = ProtectedMembersArray;
  }

  if (!I.Bases.empty()) {
    json::Value BasesArray = Array();
    json::Array &BasesArrayRef = *BasesArray.getAsArray();
    BasesArrayRef.reserve(I.Bases.size());
    for (const auto &BaseInfo : I.Bases) {
      json::Value BaseInfoVal = Object();
      auto &BaseInfoObj = *BaseInfoVal.getAsObject();
      serializeInfo(BaseInfo, BaseInfoObj, RepositoryUrl);
      BaseInfoObj["IsVirtual"] = BaseInfo.IsVirtual;
      BaseInfoObj["Access"] = getAccessSpelling(BaseInfo.Access);
      BaseInfoObj["IsParent"] = BaseInfo.IsParent;
      BasesArrayRef.push_back(BaseInfoVal);
    }
    Obj["Bases"] = BasesArray;
  }

  if (!I.Parents.empty())
    serializeReference(I.Parents, Obj, "Parents");

  if (!I.VirtualParents.empty())
    serializeReference(I.VirtualParents, Obj, "VirtualParents");

  if (I.Template)
    serializeInfo(I.Template.value(), Obj);

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
