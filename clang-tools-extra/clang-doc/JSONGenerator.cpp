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

static void serializeInfo(const ConstraintInfo &I, Object &Obj);
static void serializeInfo(const RecordInfo &I, Object &Obj,
                          const std::optional<StringRef> &RepositoryUrl);

static void serializeReference(const Reference &Ref, Object &ReferenceObj);

template <typename Container, typename SerializationFunc>
static void serializeArray(const Container &Records, Object &Obj,
                           const std::string &Key,
                           SerializationFunc SerializeInfo);

// Convenience lambda to pass to serializeArray.
// If a serializeInfo needs a RepositoryUrl, create a local lambda that captures
// the optional.
static auto SerializeInfoLambda = [](const auto &Info, Object &Object) {
  serializeInfo(Info, Object);
};
static auto SerializeReferenceLambda = [](const auto &Ref, Object &Object) {
  serializeReference(Ref, Object);
};

static std::string infoTypeToString(InfoType IT) {
  switch (IT) {
  case InfoType::IT_default:
    return "default";
  case InfoType::IT_namespace:
    return "namespace";
  case InfoType::IT_record:
    return "record";
  case InfoType::IT_function:
    return "function";
  case InfoType::IT_enum:
    return "enum";
  case InfoType::IT_typedef:
    return "typedef";
  case InfoType::IT_concept:
    return "concept";
  case InfoType::IT_variable:
    return "variable";
  case InfoType::IT_friend:
    return "friend";
  }
  llvm_unreachable("Unknown InfoType encountered.");
}

static json::Object
serializeLocation(const Location &Loc,
                  const std::optional<StringRef> RepositoryUrl) {
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

static void insertComment(Object &Description, json::Value &Comment,
                          StringRef Key) {
  auto DescriptionIt = Description.find(Key);

  if (DescriptionIt == Description.end()) {
    auto CommentsArray = json::Array();
    CommentsArray.push_back(Comment);
    Description[Key] = std::move(CommentsArray);
    Description["Has" + Key.str()] = true;
  } else {
    DescriptionIt->getSecond().getAsArray()->push_back(Comment);
  }
}

static json::Value extractTextComments(Object *ParagraphComment) {
  if (!ParagraphComment)
    return json::Object();
  return *ParagraphComment->get("Children");
}

static json::Value extractVerbatimComments(json::Array VerbatimLines) {
  json::Value TextArray = json::Array();
  auto &TextArrayRef = *TextArray.getAsArray();
  for (auto &Line : VerbatimLines)
    TextArrayRef.push_back(*Line.getAsObject()
                                ->get("VerbatimBlockLineComment")
                                ->getAsObject()
                                ->get("Text"));

  return TextArray;
}

static Object serializeComment(const CommentInfo &I, Object &Description) {
  // taken from PR #142273
  Object Obj = Object();

  json::Value ChildVal = Object();
  Object &Child = *ChildVal.getAsObject();

  json::Value ChildArr = Array();
  auto &CARef = *ChildArr.getAsArray();
  CARef.reserve(I.Children.size());
  for (const auto &C : I.Children)
    CARef.emplace_back(serializeComment(*C, Description));

  switch (I.Kind) {
  case CommentKind::CK_TextComment: {
    Obj.insert({commentKindToString(I.Kind), I.Text});
    return Obj;
  }

  case CommentKind::CK_BlockCommandComment: {
    auto TextCommentsArray = extractTextComments(CARef.front().getAsObject());
    if (I.Name == "brief")
      insertComment(Description, TextCommentsArray, "BriefComments");
    else if (I.Name == "return")
      insertComment(Description, TextCommentsArray, "ReturnComments");
    return Obj;
  }

  case CommentKind::CK_InlineCommandComment: {
    json::Value ArgsArr = Array();
    auto &ARef = *ArgsArr.getAsArray();
    ARef.reserve(I.Args.size());
    for (const auto &Arg : I.Args)
      ARef.emplace_back(Arg);
    Child.insert({"Command", I.Name});
    Child.insert({"Args", ArgsArr});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_ParamCommandComment:
  case CommentKind::CK_TParamCommandComment: {
    Child.insert({"ParamName", I.ParamName});
    Child.insert({"Direction", I.Direction});
    Child.insert({"Explicit", I.Explicit});
    auto TextCommentsArray = extractTextComments(CARef.front().getAsObject());
    Child.insert({"Children", TextCommentsArray});
    if (I.Kind == CommentKind::CK_ParamCommandComment)
      insertComment(Description, ChildVal, "ParamComments");
    return Obj;
  }

  case CommentKind::CK_VerbatimBlockComment: {
    if (I.CloseName == "endcode") {
      // We don't support \code language specification
      auto TextCommentsArray = extractVerbatimComments(CARef);
      insertComment(Description, TextCommentsArray, "CodeComments");
    } else if (I.CloseName == "endverbatim")
      insertComment(Description, ChildVal, "VerbatimComments");
    return Obj;
  }

  case CommentKind::CK_VerbatimBlockLineComment:
  case CommentKind::CK_VerbatimLineComment: {
    Child.insert({"Text", I.Text});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_HTMLStartTagComment: {
    json::Value AttrKeysArray = json::Array();
    json::Value AttrValuesArray = json::Array();
    auto &KeyArr = *AttrKeysArray.getAsArray();
    auto &ValArr = *AttrValuesArray.getAsArray();
    KeyArr.reserve(I.AttrKeys.size());
    ValArr.reserve(I.AttrValues.size());
    for (const auto &K : I.AttrKeys)
      KeyArr.emplace_back(K);
    for (const auto &V : I.AttrValues)
      ValArr.emplace_back(V);
    Child.insert({"Name", I.Name});
    Child.insert({"SelfClosing", I.SelfClosing});
    Child.insert({"AttrKeys", AttrKeysArray});
    Child.insert({"AttrValues", AttrValuesArray});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_HTMLEndTagComment: {
    Child.insert({"Name", I.Name});
    Child.insert({"Children", ChildArr});
    Obj.insert({commentKindToString(I.Kind), ChildVal});
    return Obj;
  }

  case CommentKind::CK_FullComment:
  case CommentKind::CK_ParagraphComment: {
    Child.insert({"Children", ChildArr});
    Child["ParagraphComment"] = true;
    return Child;
  }

  case CommentKind::CK_Unknown: {
    Obj.insert({commentKindToString(I.Kind), I.Text});
    return Obj;
  }
  }
  llvm_unreachable("Unknown comment kind encountered.");
}

static void
serializeCommonAttributes(const Info &I, json::Object &Obj,
                          const std::optional<StringRef> RepositoryUrl) {
  Obj["Name"] = I.Name;
  Obj["USR"] = toHex(toStringRef(I.USR));
  Obj["InfoType"] = infoTypeToString(I.IT);
  if (!I.DocumentationFileName.empty())
    Obj["DocumentationFileName"] = I.DocumentationFileName;

  if (!I.Path.empty())
    Obj["Path"] = I.Path;

  if (!I.Namespace.empty()) {
    Obj["Namespace"] = json::Array();
    for (const auto &NS : I.Namespace)
      Obj["Namespace"].getAsArray()->push_back(NS.Name);
  }

  if (!I.Description.empty()) {
    Object Description = Object();
    // Skip straight to the FullComment's children
    auto &Comments = I.Description.at(0).Children;
    for (const auto &CommentInfo : Comments) {
      json::Value Comment = serializeComment(*CommentInfo, Description);
      // if a ParagraphComment is returned, then it is a top-level comment that
      // needs to be inserted manually.
      if (auto *ParagraphComment = Comment.getAsObject();
          ParagraphComment->get("ParagraphComment")) {
        auto TextCommentsArray = extractTextComments(ParagraphComment);
        insertComment(Description, TextCommentsArray, "ParagraphComments");
      }
    }
    Obj["Description"] = std::move(Description);
  }

  // Namespaces aren't SymbolInfos, so they dont have a DefLoc
  if (I.IT != InfoType::IT_namespace) {
    const auto *Symbol = static_cast<const SymbolInfo *>(&I);
    if (Symbol->DefLoc)
      Obj["Location"] =
          serializeLocation(Symbol->DefLoc.value(), RepositoryUrl);
  }
}

static void serializeReference(const Reference &Ref, Object &ReferenceObj) {
  ReferenceObj["Path"] = Ref.Path;
  ReferenceObj["Name"] = Ref.Name;
  ReferenceObj["QualName"] = Ref.QualName;
  ReferenceObj["USR"] = toHex(toStringRef(Ref.USR));
  if (!Ref.DocumentationFileName.empty())
    ReferenceObj["DocumentationFileName"] = Ref.DocumentationFileName;
}

// Although namespaces and records both have ScopeChildren, they serialize them
// differently. Only enums, records, and typedefs are handled here.
static void
serializeCommonChildren(const ScopeChildren &Children, json::Object &Obj,
                        const std::optional<StringRef> RepositoryUrl) {
  static auto SerializeInfo = [RepositoryUrl](const auto &Info,
                                              Object &Object) {
    serializeInfo(Info, Object, RepositoryUrl);
  };

  if (!Children.Enums.empty()) {
    serializeArray(Children.Enums, Obj, "Enums", SerializeInfo);
    Obj["HasEnums"] = true;
  }

  if (!Children.Typedefs.empty())
    serializeArray(Children.Typedefs, Obj, "Typedefs", SerializeInfo);

  if (!Children.Records.empty()) {
    serializeArray(Children.Records, Obj, "Records", SerializeReferenceLambda);
    Obj["HasRecords"] = true;
  }
}

template <typename Container, typename SerializationFunc>
static void serializeArray(const Container &Records, Object &Obj,
                           const std::string &Key,
                           SerializationFunc SerializeInfo) {
  json::Value RecordsArray = Array();
  auto &RecordsArrayRef = *RecordsArray.getAsArray();
  RecordsArrayRef.reserve(Records.size());
  for (size_t Index = 0; Index < Records.size(); ++Index) {
    json::Value ItemVal = Object();
    auto &ItemObj = *ItemVal.getAsObject();
    SerializeInfo(Records[Index], ItemObj);
    if (Index == Records.size() - 1)
      ItemObj["End"] = true;
    RecordsArrayRef.push_back(ItemVal);
  }
  Obj[Key] = RecordsArray;
}

static void serializeInfo(const ConstraintInfo &I, Object &Obj) {
  serializeReference(I.ConceptRef, Obj);
  Obj["Expression"] = I.ConstraintExpr;
}

static void serializeInfo(const ArrayRef<TemplateParamInfo> &Params,
                          Object &Obj) {
  json::Value ParamsArray = Array();
  auto &ParamsArrayRef = *ParamsArray.getAsArray();
  ParamsArrayRef.reserve(Params.size());
  for (const auto &Param : Params)
    ParamsArrayRef.push_back(Param.Contents);
  Obj["Parameters"] = ParamsArray;
}

static void serializeInfo(const TemplateInfo &Template, Object &Obj) {
  json::Value TemplateVal = Object();
  auto &TemplateObj = *TemplateVal.getAsObject();

  if (Template.Specialization) {
    json::Value TemplateSpecializationVal = Object();
    auto &TemplateSpecializationObj = *TemplateSpecializationVal.getAsObject();
    TemplateSpecializationObj["SpecializationOf"] =
        toHex(toStringRef(Template.Specialization->SpecializationOf));
    if (!Template.Specialization->Params.empty())
      serializeInfo(Template.Specialization->Params, TemplateSpecializationObj);
    TemplateObj["Specialization"] = TemplateSpecializationVal;
  }

  if (!Template.Params.empty())
    serializeInfo(Template.Params, TemplateObj);

  if (!Template.Constraints.empty())
    serializeArray(Template.Constraints, TemplateObj, "Constraints",
                   SerializeInfoLambda);

  Obj["Template"] = TemplateVal;
}

static void serializeInfo(const ConceptInfo &I, Object &Obj,
                          const std::optional<StringRef> &RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["IsType"] = I.IsType;
  Obj["ConstraintExpression"] = I.ConstraintExpression;
  serializeInfo(I.Template, Obj);
}

static void serializeInfo(const TypeInfo &I, Object &Obj) {
  Obj["Name"] = I.Type.Name;
  Obj["QualName"] = I.Type.QualName;
  Obj["USR"] = toHex(toStringRef(I.Type.USR));
  Obj["IsTemplate"] = I.IsTemplate;
  Obj["IsBuiltIn"] = I.IsBuiltIn;
}

static void serializeInfo(const FieldTypeInfo &I, Object &Obj) {
  Obj["Name"] = I.Name;
  Obj["Type"] = I.Type.Name;
}

static void serializeInfo(const FunctionInfo &F, json::Object &Obj,
                          const std::optional<StringRef> RepositoryURL) {
  serializeCommonAttributes(F, Obj, RepositoryURL);
  Obj["IsStatic"] = F.IsStatic;

  auto ReturnTypeObj = Object();
  serializeInfo(F.ReturnType, ReturnTypeObj);
  Obj["ReturnType"] = std::move(ReturnTypeObj);

  if (!F.Params.empty())
    serializeArray(F.Params, Obj, "Params", SerializeInfoLambda);

  if (F.Template)
    serializeInfo(F.Template.value(), Obj);
}

static void serializeInfo(const EnumValueInfo &I, Object &Obj) {
  Obj["Name"] = I.Name;
  if (!I.ValueExpr.empty())
    Obj["ValueExpr"] = I.ValueExpr;
  else
    Obj["Value"] = I.Value;
}

static void serializeInfo(const EnumInfo &I, json::Object &Obj,
                          const std::optional<StringRef> &RepositoryUrl) {
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

  if (!I.Members.empty())
    serializeArray(I.Members, Obj, "Members", SerializeInfoLambda);
}

static void serializeInfo(const TypedefInfo &I, json::Object &Obj,
                          const std::optional<StringRef> &RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["TypeDeclaration"] = I.TypeDeclaration;
  Obj["IsUsing"] = I.IsUsing;
  json::Value TypeVal = Object();
  auto &TypeObj = *TypeVal.getAsObject();
  serializeInfo(I.Underlying, TypeObj);
  Obj["Underlying"] = TypeVal;
}

static void serializeInfo(const BaseRecordInfo &I, Object &Obj,
                          const std::optional<StringRef> &RepositoryUrl) {
  serializeInfo(static_cast<const RecordInfo &>(I), Obj, RepositoryUrl);
  Obj["IsVirtual"] = I.IsVirtual;
  Obj["Access"] = getAccessSpelling(I.Access);
  Obj["IsParent"] = I.IsParent;
}

static void serializeInfo(const FriendInfo &I, Object &Obj) {
  auto FriendRef = Object();
  serializeReference(I.Ref, FriendRef);
  Obj["Reference"] = std::move(FriendRef);
  Obj["IsClass"] = I.IsClass;
  if (I.Template)
    serializeInfo(I.Template.value(), Obj);
  if (I.Params)
    serializeArray(I.Params.value(), Obj, "Params", SerializeInfoLambda);
  if (I.ReturnType) {
    auto ReturnTypeObj = Object();
    serializeInfo(I.ReturnType.value(), ReturnTypeObj);
    Obj["ReturnType"] = std::move(ReturnTypeObj);
  }
}

static void insertArray(Object &Obj, json::Value &Array, StringRef Key) {
  Obj[Key] = Array;
  Obj["Has" + Key.str()] = true;
}

static void serializeInfo(const RecordInfo &I, json::Object &Obj,
                          const std::optional<StringRef> &RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["FullName"] = I.FullName;
  Obj["TagType"] = getTagType(I.TagType);
  Obj["IsTypedef"] = I.IsTypeDef;
  Obj["MangledName"] = I.MangledName;

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
      insertArray(Obj, PubFunctionsArray, "PublicFunctions");
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
      insertArray(Obj, PublicMembersArray, "PublicMembers");
    if (!ProtMembersArrayRef.empty())
      Obj["ProtectedMembers"] = ProtectedMembersArray;
  }

  if (!I.Bases.empty())
    serializeArray(
        I.Bases, Obj, "Bases",
        [&RepositoryUrl](const BaseRecordInfo &Base, Object &BaseObj) {
          serializeInfo(Base, BaseObj, RepositoryUrl);
        });

  if (!I.Parents.empty())
    serializeArray(I.Parents, Obj, "Parents", SerializeReferenceLambda);

  if (!I.VirtualParents.empty())
    serializeArray(I.VirtualParents, Obj, "VirtualParents",
                   SerializeReferenceLambda);

  if (I.Template)
    serializeInfo(I.Template.value(), Obj);

  if (!I.Friends.empty())
    serializeArray(I.Friends, Obj, "Friends", SerializeInfoLambda);

  serializeCommonChildren(I.Children, Obj, RepositoryUrl);
}

static void serializeInfo(const VarInfo &I, json::Object &Obj,
                          const std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);
  Obj["IsStatic"] = I.IsStatic;
  auto TypeObj = Object();
  serializeInfo(I.Type, TypeObj);
  Obj["Type"] = std::move(TypeObj);
}

static void serializeInfo(const NamespaceInfo &I, json::Object &Obj,
                          const std::optional<StringRef> RepositoryUrl) {
  serializeCommonAttributes(I, Obj, RepositoryUrl);

  if (!I.Children.Namespaces.empty())
    serializeArray(I.Children.Namespaces, Obj, "Namespaces",
                   SerializeReferenceLambda);

  static auto SerializeInfo = [RepositoryUrl](const auto &Info,
                                              Object &Object) {
    serializeInfo(Info, Object, RepositoryUrl);
  };

  if (!I.Children.Functions.empty())
    serializeArray(I.Children.Functions, Obj, "Functions", SerializeInfo);

  if (!I.Children.Concepts.empty())
    serializeArray(I.Children.Concepts, Obj, "Concepts", SerializeInfo);

  if (!I.Children.Variables.empty())
    serializeArray(I.Children.Variables, Obj, "Variables", SerializeInfo);

  serializeCommonChildren(I.Children, Obj, RepositoryUrl);
}

static SmallString<16> determineFileName(Info *I, SmallString<128> &Path) {
  SmallString<16> FileName;
  if (I->IT == InfoType::IT_record) {
    auto *RecordSymbolInfo = static_cast<SymbolInfo *>(I);
    FileName = RecordSymbolInfo->MangledName;
  } else if (I->USR == GlobalNamespaceID)
    FileName = "index";
  else if (I->IT == InfoType::IT_namespace) {
    for (const auto &NS : I->Namespace) {
      FileName += NS.Name;
      FileName += "_";
    }
    FileName += I->Name;
  } else
    FileName = I->Name;
  sys::path::append(Path, FileName + ".json");
  return FileName;
}

// FIXME: Revert back to creating nested directories for namespaces instead of
// putting everything in a flat directory structure.
Error JSONGenerator::generateDocs(
    StringRef RootDir, llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
    const ClangDocContext &CDCtx) {
  StringSet<> CreatedDirs;
  StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    Info *Info = Group.getValue().get();

    SmallString<128> Path;
    auto RootDirStr = RootDir.str() + "/json";
    StringRef JSONDir = StringRef(RootDirStr);
    sys::path::native(JSONDir, Path);
    if (!CreatedDirs.contains(Path)) {
      if (std::error_code Err = sys::fs::create_directories(Path);
          Err != std::error_code())
        return createFileError(Twine(Path), Err);
      CreatedDirs.insert(Path);
    }

    SmallString<16> FileName = determineFileName(Info, Path);
    if (FileToInfos.contains(Path))
      continue;
    FileToInfos[Path].push_back(Info);
    Info->DocumentationFileName = FileName;
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
    serializeInfo(*static_cast<NamespaceInfo *>(I), Obj, CDCtx.RepositoryUrl);
    break;
  case InfoType::IT_record:
    serializeInfo(*static_cast<RecordInfo *>(I), Obj, CDCtx.RepositoryUrl);
    break;
  case InfoType::IT_concept:
  case InfoType::IT_enum:
  case InfoType::IT_function:
  case InfoType::IT_typedef:
  case InfoType::IT_variable:
  case InfoType::IT_friend:
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
