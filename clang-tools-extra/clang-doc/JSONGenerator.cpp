#include "Generators.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::json;

namespace clang {
namespace doc {

template <typename Container, typename SerializationFunc>
static void serializeArray(
    const Container &Records, Object &Obj, const StringRef Key,
    SerializationFunc SerializeInfo, const StringRef EndKey = "End",
    function_ref<void(Object &)> UpdateJson = [](Object &Obj) {});

// TODO(issue URL): Wrapping logic for HTML should probably use a more
// sophisticated heuristic than number of parameters.
constexpr static unsigned getMaxParamWrapLimit() { return 2; }

typedef std::function<void(const Reference &, Object &)> ReferenceFunc;

class JSONGenerator : public Generator {
  json::Object serializeLocation(const Location &Loc);
  void serializeCommonAttributes(const Info &I, json::Object &Obj);
  void serializeCommonChildren(
      const ScopeChildren &Children, json::Object &Obj,
      std::optional<ReferenceFunc> MDReferenceLambda = std::nullopt);
  void serializeInfo(const ConstraintInfo &I, Object &Obj);
  void serializeInfo(const TemplateInfo &Template, Object &Obj);
  void serializeInfo(const ConceptInfo &I, Object &Obj);
  void serializeInfo(const TypeInfo &I, Object &Obj);
  void serializeInfo(const FieldTypeInfo &I, Object &Obj);
  void serializeInfo(const FunctionInfo &F, json::Object &Obj);
  void serializeInfo(const EnumValueInfo &I, Object &Obj);
  void serializeInfo(const EnumInfo &I, json::Object &Obj);
  void serializeInfo(const TypedefInfo &I, json::Object &Obj);
  void serializeInfo(const BaseRecordInfo &I, Object &Obj);
  void serializeInfo(const FriendInfo &I, Object &Obj);
  void serializeInfo(const RecordInfo &I, json::Object &Obj);
  void serializeInfo(const VarInfo &I, json::Object &Obj);
  void serializeInfo(const NamespaceInfo &I, json::Object &Obj);
  SmallString<16> determineFileName(Info *I, SmallString<128> &Path);
  Error serializeIndex(StringRef RootDir);
  void generateContext(const Info &I, Object &Obj);
  void serializeReference(const Reference &Ref, Object &ReferenceObj);
  Error serializeAllFiles(const ClangDocContext &CDCtx, StringRef RootDir);
  void serializeMDReference(const Reference &Ref, Object &ReferenceObj,
                            StringRef BasePath);

  // Convenience lambdas to pass to serializeArray.
  auto serializeInfoLambda() {
    return [this](const auto &Info, Object &Object) {
      serializeInfo(Info, Object);
    };
  }
  auto serializeReferenceLambda() {
    return [this](const auto &Ref, Object &Object) {
      serializeReference(Ref, Object);
    };
  }

public:
  static const char *Format;
  const ClangDocContext *CDCtx;
  bool Markdown;

  Error generateDocumentation(StringRef RootDir,
                              llvm::StringMap<OwnedPtr<doc::Info>> Infos,
                              const ClangDocContext &CDCtx,
                              std::string DirName) override;
  Error createResources(ClangDocContext &CDCtx) override;
  // FIXME: Once legacy generators are removed, we can refactor the Generator
  // interface to sto passing CDCtx here since we hold a pointer to it.
  Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                           const ClangDocContext &CDCtx) override;
};

const char *JSONGenerator::Format = "json";

static void insertNonEmpty(StringRef Key, StringRef Value, Object &Obj) {
  if (!Value.empty())
    Obj[Key] = Value;
}

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

json::Object JSONGenerator::serializeLocation(const Location &Loc) {
  Object LocationObj = Object();
  LocationObj["LineNumber"] = Loc.StartLineNumber;
  LocationObj["Filename"] = Loc.Filename;

  if (!Loc.IsFileInRootDir || !CDCtx->RepositoryUrl)
    return LocationObj;
  SmallString<128> FileURL(*CDCtx->RepositoryUrl);
  sys::path::append(FileURL, sys::path::Style::posix, Loc.Filename);

  std::string LinePrefix;
  if (!CDCtx->RepositoryLinePrefix)
    LinePrefix = "#L";
  else
    LinePrefix = *CDCtx->RepositoryLinePrefix;

  FileURL += LinePrefix + std::to_string(Loc.StartLineNumber);
  LocationObj["FileURL"] = FileURL;
  return LocationObj;
}

/// Insert comments into a key in the Description object.
///
/// \param Comment Either an Object or Array, depending on the comment type
/// \param Key     The type (Brief, Code, etc.) of comment to be inserted
static void insertComment(Object &Description, json::Value &Comment,
                          StringRef Key) {
  // The comment has a Children array for the actual text, with meta attributes
  // alongside it in the Object.
  if (auto *Obj = Comment.getAsObject()) {
    if (auto *Children = Obj->getArray("Children");
        Children && Children->empty())
      return;
  }
  // The comment is just an array of text comments.
  else if (auto *Array = Comment.getAsArray(); Array && Array->empty()) {
    return;
  }

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

/// Takes the nested "Children" array from a comment Object.
///
/// \return a json::Array of comments, possible json::Value::Kind::Null
static json::Value extractTextComments(Object *ParagraphComment) {
  if (!ParagraphComment)
    return json::Value(nullptr);
  json::Value *Children = ParagraphComment->get("Children");
  if (!Children)
    return json::Value(nullptr);
  auto ChildrenArray = *Children->getAsArray();
  auto ChildrenIt = ChildrenArray.begin();
  while (ChildrenIt != ChildrenArray.end()) {
    auto *ChildObj = ChildrenIt->getAsObject();
    assert(ChildObj && "Invalid JSON object in Comment");
    auto TextComment = ChildObj->getString("TextComment");
    if (!TextComment || TextComment->empty()) {
      ChildrenIt = ChildrenArray.erase(ChildrenIt);
      continue;
    }
    ++ChildrenIt;
  }
  return ChildrenArray;
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
    if (!I.Text.empty())
      Obj.insert({commentKindToString(I.Kind), I.Text});
    return Obj;
  }

  case CommentKind::CK_BlockCommandComment: {
    auto TextCommentsArray = extractTextComments(CARef.front().getAsObject());
    if (I.Name == "brief")
      insertComment(Description, TextCommentsArray, "BriefComments");
    else if (I.Name == "return")
      insertComment(Description, TextCommentsArray, "ReturnComments");
    else if (I.Name == "throws" || I.Name == "throw") {
      json::Value ThrowsVal = Object();
      auto &ThrowsObj = *ThrowsVal.getAsObject();
      ThrowsObj["Exception"] = I.Args.front();
      ThrowsObj["Children"] = TextCommentsArray;
      insertComment(Description, ThrowsVal, "ThrowsComments");
    }
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
    if (I.Kind == CommentKind::CK_TParamCommandComment)
      insertComment(Description, ChildVal, "TParamComments");
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

/// Creates Contexts for namespaces and records to allow for navigation.
void JSONGenerator::generateContext(const Info &I, Object &Obj) {
  json::Value ContextArray = json::Array();
  auto &ContextArrayRef = *ContextArray.getAsArray();
  ContextArrayRef.reserve(I.Contexts.size());

  std::string CurrentRelativePath;
  bool PreviousRecord = false;
  for (const auto &Current : I.Contexts) {
    json::Value ContextVal = Object();
    Object &Context = *ContextVal.getAsObject();
    serializeReference(Current, Context);

    if (ContextArrayRef.empty() && I.IT == InfoType::IT_record) {
      if (Current.DocumentationFileName == "index") {
        // If the record's immediate context is a namespace, then the
        // "index.html" is in the same directory.
        PreviousRecord = false;
        Context["RelativePath"] = "./";
      } else {
        // If the immediate context is a record, then the file is one level
        // above
        PreviousRecord = true;
        CurrentRelativePath += "../";
        Context["RelativePath"] = CurrentRelativePath;
      }
      ContextArrayRef.push_back(ContextVal);
      continue;
    }

    if (PreviousRecord && (Current.DocumentationFileName == "index")) {
      // If the previous Context was a record then we already went up a level,
      // so the current namespace index is in the same directory.
      PreviousRecord = false;
    } else if (Current.DocumentationFileName != "index") {
      // If the current Context is a record but the previous wasn't a record,
      // then the namespace index is located one level above.
      PreviousRecord = true;
      CurrentRelativePath += "../";
    } else {
      // The current Context is a namespace and so was the previous Context.
      PreviousRecord = false;
      CurrentRelativePath += "../";
      // If this namespace is the global namespace, then its documentation
      // name needs to be changed to link correctly.
      if (Current.QualName == "GlobalNamespace" && Current.RelativePath != "./")
        Context["DocumentationFileName"] =
            SmallString<16>("GlobalNamespace/index");
    }
    Context["RelativePath"] = CurrentRelativePath;
    ContextArrayRef.insert(ContextArrayRef.begin(), ContextVal);
  }

  ContextArrayRef.back().getAsObject()->insert({"End", true});
  Obj["Contexts"] = ContextArray;
  Obj["HasContexts"] = true;
}

static void serializeDescription(llvm::ArrayRef<CommentInfo> Description,
                                 json::Object &Obj, StringRef Key = "") {
  if (Description.empty())
    return;

  // Skip straight to the FullComment's children
  auto &Comments = Description.front().Children;
  Object DescriptionObj = Object();
  for (const auto &CommentInfo : Comments) {
    json::Value Comment = serializeComment(*CommentInfo, DescriptionObj);
    // if a ParagraphComment is returned, then it is a top-level comment that
    // needs to be inserted manually.
    if (auto *ParagraphComment = Comment.getAsObject();
        ParagraphComment->get("ParagraphComment")) {
      auto TextCommentsArray = extractTextComments(ParagraphComment);
      if (TextCommentsArray.kind() == json::Value::Null ||
          TextCommentsArray.getAsArray()->empty())
        continue;
      insertComment(DescriptionObj, TextCommentsArray, "ParagraphComments");
    }
  }
  Obj["Description"] = std::move(DescriptionObj);
  if (!Key.empty())
    Obj[Key] = true;
}

void JSONGenerator::serializeCommonAttributes(const Info &I,
                                              json::Object &Obj) {
  insertNonEmpty("Name", I.Name, Obj);
  if (!(I.USR == GlobalNamespaceID))
    Obj["USR"] = toHex(toStringRef(I.USR));
  Obj["InfoType"] = infoTypeToString(I.IT);
  // Conditionally insert fields.
  // Empty properties are omitted because Mustache templates use existence
  // to conditionally render content.
  insertNonEmpty("DocumentationFileName", I.DocumentationFileName, Obj);
  insertNonEmpty("Path", I.Path, Obj);

  if (!I.Namespace.empty()) {
    Obj["Namespace"] = json::Array();
    for (const auto &NS : I.Namespace)
      Obj["Namespace"].getAsArray()->push_back(NS.Name);
  }

  serializeDescription(I.Description, Obj);

  // Namespaces aren't SymbolInfos, so they dont have a DefLoc
  if (I.IT != InfoType::IT_namespace) {
    const auto *Symbol = static_cast<const SymbolInfo *>(&I);
    if (Symbol->DefLoc)
      Obj["Location"] = serializeLocation(Symbol->DefLoc.value());
  }

  if (!I.Contexts.empty())
    generateContext(I, Obj);
}

void JSONGenerator::serializeReference(const Reference &Ref,
                                       Object &ReferenceObj) {
  insertNonEmpty("Path", Ref.Path, ReferenceObj);
  ReferenceObj["Name"] = Ref.Name;
  ReferenceObj["QualName"] = Ref.QualName;
  ReferenceObj["USR"] = toHex(toStringRef(Ref.USR));
  if (!Ref.DocumentationFileName.empty()) {
    ReferenceObj["DocumentationFileName"] = Ref.DocumentationFileName;

    // If the reference is a nested class it will be put into a folder named
    // after the parent class. We can get that name from the path's stem.
    if (Ref.Path != "GlobalNamespace" && !Ref.Path.empty())
      ReferenceObj["PathStem"] = sys::path::stem(Ref.Path);
  }
}

void JSONGenerator::serializeMDReference(const Reference &Ref,
                                         Object &ReferenceObj,
                                         StringRef BasePath) {
  serializeReference(Ref, ReferenceObj);
  SmallString<64> Path = Ref.getRelativeFilePath(BasePath);
  sys::path::native(Path, sys::path::Style::posix);
  sys::path::append(Path, sys::path::Style::posix,
                    Ref.getFileBaseName() + ".md");
  ReferenceObj["BasePath"] = Path;
}

// Although namespaces and records both have ScopeChildren, they serialize them
// differently. Only enums, records, and typedefs are handled here.
void JSONGenerator::serializeCommonChildren(
    const ScopeChildren &Children, json::Object &Obj,
    std::optional<ReferenceFunc> MDReferenceLambda) {
  if (!Children.Enums.empty()) {
    serializeArray(Children.Enums, Obj, "Enums", serializeInfoLambda());
    Obj["HasEnums"] = true;
  }

  if (!Children.Typedefs.empty()) {
    serializeArray(Children.Typedefs, Obj, "Typedefs", serializeInfoLambda());
    Obj["HasTypedefs"] = true;
  }

  if (!Children.Records.empty()) {
    ReferenceFunc SerializeReferenceFunc = MDReferenceLambda
                                               ? MDReferenceLambda.value()
                                               : serializeReferenceLambda();
    serializeArray(Children.Records, Obj, "Records", SerializeReferenceFunc);
    Obj["HasRecords"] = true;
  }
}

template <typename Container, typename SerializationFunc>
static void serializeArray(const Container &Records, Object &Obj, StringRef Key,
                           SerializationFunc SerializeInfo, StringRef EndKey,
                           function_ref<void(Object &)> UpdateJson) {
  json::Value RecordsArray = Array();
  auto &RecordsArrayRef = *RecordsArray.getAsArray();
  RecordsArrayRef.reserve(Records.size());
  size_t Index = 0;
  size_t Size = Records.size();
  for (const auto &Item : Records) {
    json::Value ItemVal = Object();
    auto &ItemObj = *ItemVal.getAsObject();
    SerializeInfo(Item, ItemObj);
    if (Index == Size - 1)
      ItemObj[EndKey] = true;
    RecordsArrayRef.push_back(ItemVal);
    ++Index;
  }
  Obj[Key] = RecordsArray;
  UpdateJson(Obj);
}

void JSONGenerator::serializeInfo(const ConstraintInfo &I, Object &Obj) {
  serializeReference(I.ConceptRef, Obj);
  Obj["Expression"] = I.ConstraintExpr;
}

void JSONGenerator::serializeInfo(const TemplateInfo &Template, Object &Obj) {
  json::Value TemplateVal = Object();
  auto &TemplateObj = *TemplateVal.getAsObject();
  auto SerializeTemplateParam = [](const TemplateParamInfo &Param,
                                   Object &JsonObj) {
    JsonObj["Param"] = Param.Contents;
  };

  if (Template.Specialization) {
    json::Value TemplateSpecializationVal = Object();
    auto &TemplateSpecializationObj = *TemplateSpecializationVal.getAsObject();
    TemplateSpecializationObj["SpecializationOf"] =
        toHex(toStringRef(Template.Specialization->SpecializationOf));
    if (!Template.Specialization->Params.empty()) {
      bool VerticalDisplay =
          Template.Specialization->Params.size() > getMaxParamWrapLimit();
      serializeArray(Template.Specialization->Params, TemplateSpecializationObj,
                     "Parameters", SerializeTemplateParam, "SpecParamEnd",
                     [VerticalDisplay](Object &JsonObj) {
                       JsonObj["VerticalDisplay"] = VerticalDisplay;
                     });
    }
    TemplateObj["Specialization"] = TemplateSpecializationVal;
  }

  if (!Template.Params.empty()) {
    bool VerticalDisplay = Template.Params.size() > getMaxParamWrapLimit();
    serializeArray(Template.Params, TemplateObj, "Parameters",
                   SerializeTemplateParam, "End",
                   [VerticalDisplay](Object &JsonObj) {
                     JsonObj["VerticalDisplay"] = VerticalDisplay;
                   });
  }

  if (!Template.Constraints.empty())
    serializeArray(Template.Constraints, TemplateObj, "Constraints",
                   serializeInfoLambda());

  Obj["Template"] = TemplateVal;
}

void JSONGenerator::serializeInfo(const ConceptInfo &I, Object &Obj) {
  serializeCommonAttributes(I, Obj);
  Obj["IsType"] = I.IsType;
  Obj["ConstraintExpression"] = I.ConstraintExpression;
  serializeInfo(I.Template, Obj);
}

void JSONGenerator::serializeInfo(const TypeInfo &I, Object &Obj) {
  Obj["Name"] = I.Type.Name;
  Obj["QualName"] = I.Type.QualName;
  Obj["USR"] = toHex(toStringRef(I.Type.USR));
  Obj["IsTemplate"] = I.IsTemplate;
  Obj["IsBuiltIn"] = I.IsBuiltIn;
}

void JSONGenerator::serializeInfo(const FieldTypeInfo &I, Object &Obj) {
  Obj["Name"] = I.Name;
  insertNonEmpty("DefaultValue", I.DefaultValue, Obj);
  json::Value ReferenceVal = Object();
  Object &ReferenceObj = *ReferenceVal.getAsObject();
  serializeReference(I.Type, ReferenceObj);
  Obj["Type"] = ReferenceVal;
}

void JSONGenerator::serializeInfo(const FunctionInfo &F, json::Object &Obj) {
  serializeCommonAttributes(F, Obj);
  Obj["IsStatic"] = F.IsStatic;

  auto ReturnTypeObj = Object();
  serializeInfo(F.ReturnType, ReturnTypeObj);
  Obj["ReturnType"] = std::move(ReturnTypeObj);

  if (!F.Params.empty()) {
    const bool VerticalDisplay = F.Params.size() > getMaxParamWrapLimit();
    serializeArray(F.Params, Obj, "Params", serializeInfoLambda(), "ParamEnd",
                   [VerticalDisplay](Object &JsonObj) {
                     JsonObj["VerticalDisplay"] = VerticalDisplay;
                   });
  }

  if (F.Template)
    serializeInfo(F.Template.value(), Obj);
}

void JSONGenerator::serializeInfo(const EnumValueInfo &I, Object &Obj) {
  Obj["Name"] = I.Name;
  if (!I.ValueExpr.empty())
    Obj["ValueExpr"] = I.ValueExpr;
  else
    Obj["Value"] = I.Value;

  serializeDescription(I.Description, Obj, "HasEnumMemberComments");
}

void JSONGenerator::serializeInfo(const EnumInfo &I, json::Object &Obj) {
  serializeCommonAttributes(I, Obj);
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
    for (const auto &Member : I.Members) {
      if (!Member.Description.empty()) {
        Obj["HasComments"] = true;
        break;
      }
    }
    serializeArray(I.Members, Obj, "Members", serializeInfoLambda());
  }
}

void JSONGenerator::serializeInfo(const TypedefInfo &I, json::Object &Obj) {
  serializeCommonAttributes(I, Obj);
  Obj["TypeDeclaration"] = I.TypeDeclaration;
  Obj["IsUsing"] = I.IsUsing;
  json::Value TypeVal = Object();
  auto &TypeObj = *TypeVal.getAsObject();
  serializeInfo(I.Underlying, TypeObj);
  Obj["Underlying"] = TypeVal;
  if (I.Template)
    serializeInfo(I.Template.value(), Obj);
}

void JSONGenerator::serializeInfo(const BaseRecordInfo &I, Object &Obj) {
  serializeInfo(static_cast<const RecordInfo &>(I), Obj);
  Obj["IsVirtual"] = I.IsVirtual;
  Obj["Access"] = getAccessSpelling(I.Access);
  Obj["IsParent"] = I.IsParent;
}

void JSONGenerator::serializeInfo(const FriendInfo &I, Object &Obj) {
  auto FriendRef = Object();
  serializeReference(I.Ref, FriendRef);
  Obj["Reference"] = std::move(FriendRef);
  Obj["IsClass"] = I.IsClass;
  if (I.Template)
    serializeInfo(I.Template.value(), Obj);
  if (!I.Params.empty())
    serializeArray(I.Params, Obj, "Params", serializeInfoLambda());
  if (I.ReturnType) {
    auto ReturnTypeObj = Object();
    serializeInfo(I.ReturnType.value(), ReturnTypeObj);
    Obj["ReturnType"] = std::move(ReturnTypeObj);
  }
  serializeCommonAttributes(I, Obj);
}

static void insertArray(Object &Obj, json::Value &Array, StringRef Key) {
  Obj[Key] = Array;
  Obj["Has" + Key.str()] = true;
}

void JSONGenerator::serializeInfo(const RecordInfo &I, json::Object &Obj) {
  serializeCommonAttributes(I, Obj);
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
      serializeInfo(Function, FunctionObj);
      AccessSpecifier Access = Function.Access;
      if (Access == AccessSpecifier::AS_public)
        PubFunctionsArrayRef.push_back(FunctionVal);
      else if (Access == AccessSpecifier::AS_protected)
        ProtFunctionsArrayRef.push_back(FunctionVal);
    }

    if (!PubFunctionsArrayRef.empty())
      insertArray(Obj, PubFunctionsArray, "PublicMethods");
    if (!ProtFunctionsArrayRef.empty())
      insertArray(Obj, ProtFunctionsArray, "ProtectedMethods");
  }

  if (!I.Members.empty()) {
    Obj["HasMembers"] = true;
    json::Value PublicMembersArray = Array();
    json::Array &PubMembersArrayRef = *PublicMembersArray.getAsArray();
    json::Value ProtectedMembersArray = Array();
    json::Array &ProtMembersArrayRef = *ProtectedMembersArray.getAsArray();
    json::Value PrivateMembersArray = Array();
    json::Array &PrivateMembersArrayRef = *PrivateMembersArray.getAsArray();

    for (const MemberTypeInfo &Member : I.Members) {
      json::Value MemberVal = Object();
      auto &MemberObj = *MemberVal.getAsObject();
      MemberObj["Name"] = Member.Name;
      MemberObj["Type"] = Member.Type.Name;
      MemberObj["IsStatic"] = Member.IsStatic;

      if (Member.Access == AccessSpecifier::AS_public)
        PubMembersArrayRef.push_back(MemberVal);
      else if (Member.Access == AccessSpecifier::AS_protected)
        ProtMembersArrayRef.push_back(MemberVal);
      else if (Member.Access == AccessSpecifier::AS_private)
        PrivateMembersArrayRef.push_back(MemberVal);
    }

    if (!PubMembersArrayRef.empty())
      insertArray(Obj, PublicMembersArray, "PublicMembers");
    if (!ProtMembersArrayRef.empty())
      insertArray(Obj, ProtectedMembersArray, "ProtectedMembers");
    if (!PrivateMembersArrayRef.empty())
      insertArray(Obj, PrivateMembersArray, "PrivateMembers");
  }

  if (!I.Bases.empty())
    serializeArray(I.Bases, Obj, "Bases", serializeInfoLambda());

  if (!I.Parents.empty()) {
    serializeArray(I.Parents, Obj, "Parents", serializeReferenceLambda());
    Obj["HasParents"] = true;
  }

  if (!I.VirtualParents.empty()) {
    serializeArray(I.VirtualParents, Obj, "VirtualParents",
                   serializeReferenceLambda());
    Obj["HasVirtualParents"] = true;
  }

  if (I.Template)
    serializeInfo(I.Template.value(), Obj);

  if (!I.Friends.empty()) {
    serializeArray(I.Friends, Obj, "Friends", serializeInfoLambda());
    Obj["HasFriends"] = true;
  }

  serializeCommonChildren(I.Children, Obj);
}

void JSONGenerator::serializeInfo(const VarInfo &I, json::Object &Obj) {
  serializeCommonAttributes(I, Obj);
  Obj["IsStatic"] = I.IsStatic;
  auto TypeObj = Object();
  serializeInfo(I.Type, TypeObj);
  Obj["Type"] = std::move(TypeObj);
}

void JSONGenerator::serializeInfo(const NamespaceInfo &I, json::Object &Obj) {
  serializeCommonAttributes(I, Obj);
  if (I.USR == GlobalNamespaceID)
    Obj["Name"] = "Global Namespace";

  if (!I.Children.Functions.empty()) {
    serializeArray(I.Children.Functions, Obj, "Functions",
                   serializeInfoLambda());
    Obj["HasFunctions"] = true;
  }

  if (!I.Children.Concepts.empty()) {
    serializeArray(I.Children.Concepts, Obj, "Concepts", serializeInfoLambda());
    Obj["HasConcepts"] = true;
  }

  if (!I.Children.Variables.empty()) {
    serializeArray(I.Children.Variables, Obj, "Variables",
                   serializeInfoLambda());
    Obj["HasVariables"] = true;
  }

  ReferenceFunc SerializeReferenceFunc;
  if (Markdown) {
    SmallString<64> BasePath = I.getRelativeFilePath("");
    // serializeCommonChildren doesn't accept Infos, so this lambda needs to be
    // created here. To avoid making serializeCommonChildren a template, this
    // lambda is an std::function
    SerializeReferenceFunc = [this, BasePath](const Reference &Ref,
                                              Object &Object) {
      serializeMDReference(Ref, Object, BasePath);
    };
    serializeCommonChildren(I.Children, Obj, SerializeReferenceFunc);
  } else {
    SerializeReferenceFunc = serializeReferenceLambda();
    serializeCommonChildren(I.Children, Obj);
  }

  if (!I.Children.Namespaces.empty()) {
    serializeArray(I.Children.Namespaces, Obj, "Namespaces",
                   SerializeReferenceFunc);
    Obj["HasNamespaces"] = true;
  }
}

SmallString<16> JSONGenerator::determineFileName(Info *I,
                                                 SmallString<128> &Path) {
  SmallString<16> FileName;
  if (I->IT == InfoType::IT_record) {
    auto *RecordSymbolInfo = static_cast<SymbolInfo *>(I);
    FileName = RecordSymbolInfo->MangledName;
  } else if (I->IT == InfoType::IT_namespace) {
    FileName = "index";
  } else
    FileName = I->Name;
  sys::path::append(Path, FileName + ".json");
  return FileName;
}

/// \param CDCtxIndex Passed by copy since clang-doc's context is passed to the
/// generator as `const`
static OwningVec<Index> preprocessCDCtxIndex(Index CDCtxIndex) {
  CDCtxIndex.sort();
  OwningVec<Index> Processed;
  Processed.reserve(CDCtxIndex.Children.size());
  for (const auto *Idx : CDCtxIndex.getSortedChildren()) {
    Index NewIdx = *Idx;
    SmallString<128> NewPath(NewIdx.getRelativeFilePath(""));
    sys::path::native(NewPath, sys::path::Style::posix);
    sys::path::append(NewPath, sys::path::Style::posix,
                      NewIdx.getFileBaseName() + ".md");
    NewIdx.Path = internString(NewPath);
    Processed.push_back(NewIdx);
  }

  return Processed;
}

/// Serialize ClangDocContext's Index for Markdown output
Error JSONGenerator::serializeAllFiles(const ClangDocContext &CDCtx,
                                       StringRef RootDir) {
  json::Value ObjVal = Object();
  Object &Obj = *ObjVal.getAsObject();
  OwningVec<Index> IndexCopy = preprocessCDCtxIndex(CDCtx.Idx);
  serializeArray(IndexCopy, Obj, "Index", serializeReferenceLambda());
  SmallString<128> Path;
  sys::path::append(Path, RootDir, "json", "all_files.json");
  std::error_code FileErr;
  raw_fd_ostream RootOS(Path, FileErr, sys::fs::OF_Text);
  if (FileErr)
    return createFileError("cannot open file " + Path, FileErr);
  RootOS << llvm::formatv("{0:2}", ObjVal);
  return Error::success();
}

// Creates a JSON file above the global namespace directory.
// An index can be used to create the top-level HTML index page or the Markdown
// index file.
Error JSONGenerator::serializeIndex(StringRef RootDir) {
  if (CDCtx->Idx.Children.empty())
    return Error::success();

  json::Value ObjVal = Object();
  Object &Obj = *ObjVal.getAsObject();
  insertNonEmpty("ProjectName", CDCtx->ProjectName, Obj);

  auto IndexCopy = CDCtx->Idx;
  IndexCopy.sort();
  json::Value IndexArray = json::Array();
  auto &IndexArrayRef = *IndexArray.getAsArray();

  if (IndexCopy.Children.empty()) {
    // If the index is empty, default to displaying the global namespace.
    IndexCopy.Children.try_emplace(toStringRef(GlobalNamespaceID),
                                   GlobalNamespaceID, "",
                                   InfoType::IT_namespace, "GlobalNamespace");
  } else {
    IndexArrayRef.reserve(CDCtx->Idx.Children.size());
  }

  auto Children = IndexCopy.getSortedChildren();

  for (const auto *Idx : Children) {
    if (Idx->Children.empty())
      continue;
    std::string TypeStr = infoTypeToString(Idx->RefType);
    json::Value IdxVal = Object();
    auto &IdxObj = *IdxVal.getAsObject();
    if (Markdown)
      TypeStr.at(0) = toUppercase(TypeStr.at(0));
    IdxObj["Type"] = TypeStr;
    serializeReference(*Idx, IdxObj);
    IndexArrayRef.push_back(IdxVal);
  }
  Obj["Index"] = IndexArray;

  SmallString<128> IndexFilePath(RootDir);
  sys::path::append(IndexFilePath, "/json/index.json");
  std::error_code FileErr;
  raw_fd_ostream RootOS(IndexFilePath, FileErr, sys::fs::OF_Text);
  if (FileErr)
    return createFileError("cannot open file " + IndexFilePath, FileErr);
  RootOS << llvm::formatv("{0:2}", ObjVal);
  return Error::success();
}

static void serializeContexts(Info *I, StringMap<OwnedPtr<Info>> &Infos) {
  if (I->USR == GlobalNamespaceID)
    return;
  auto ParentUSR = I->ParentUSR;

  while (true) {
    // Infos may not have the ParentUSR, if its been filtered (public or path),
    // so we can't use at() for the lookup, since it would abort.
    auto Iter = Infos.find(llvm::toHex(ParentUSR));
    if (Iter == Infos.end())
      break;
    auto &ParentInfo = Iter->second;

    if (ParentInfo && ParentInfo->USR == GlobalNamespaceID) {
      Context GlobalRef(ParentInfo->USR, "Global Namespace",
                        InfoType::IT_namespace, "GlobalNamespace", "",
                        SmallString<16>("index"));
      I->Contexts.push_back(GlobalRef);
      return;
    }

    Context ParentRef(*ParentInfo);
    I->Contexts.push_back(ParentRef);
    ParentUSR = ParentInfo->ParentUSR;
  }
}

Error JSONGenerator::generateDocumentation(
    StringRef RootDir, llvm::StringMap<doc::OwnedPtr<doc::Info>> Infos,
    const ClangDocContext &CDCtx, std::string DirName) {
  this->CDCtx = &CDCtx;
  StringSet<> CreatedDirs;
  StringMap<std::vector<doc::Info *>> FileToInfos;
  for (const auto &Group : Infos) {
    Info *Info = getPtr(Group.getValue());

    SmallString<128> Path;
    auto RootDirStr = RootDir.str() + "/json";
    StringRef JSONDir = StringRef(RootDirStr);
    sys::path::native(JSONDir, Path);
    sys::path::append(Path, Info->getRelativeFilePath(""));
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
    Info->DocumentationFileName = internString(FileName);
  }

  if (CDCtx.Format == OutputFormatTy::md_mustache) {
    Markdown = true;
    if (auto Err = serializeAllFiles(CDCtx, RootDir))
      return Err;
  }

  for (const auto &Group : FileToInfos) {
    std::error_code FileErr;
    raw_fd_ostream InfoOS(Group.getKey(), FileErr, sys::fs::OF_Text);
    if (FileErr)
      return createFileError("cannot open file " + Group.getKey(), FileErr);

    for (const auto &Info : Group.getValue()) {
      if (Info->IT == InfoType::IT_record || Info->IT == InfoType::IT_namespace)
        serializeContexts(Info, Infos);
      if (Error Err = generateDocForInfo(Info, InfoOS, CDCtx))
        return Err;
    }
  }

  return serializeIndex(RootDir);
}

Error JSONGenerator::generateDocForInfo(Info *I, raw_ostream &OS,
                                        const ClangDocContext &CDCtx) {
  json::Object Obj = Object();

  switch (I->IT) {
  case InfoType::IT_namespace:
    serializeInfo(*static_cast<NamespaceInfo *>(I), Obj);
    break;
  case InfoType::IT_record:
    serializeInfo(*static_cast<RecordInfo *>(I), Obj);
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
