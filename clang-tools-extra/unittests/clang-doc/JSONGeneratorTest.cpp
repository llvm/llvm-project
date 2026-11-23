#include "ClangDocTest.h"
#include "Generators.h"
#include "Representation.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

static std::unique_ptr<Generator> getJSONGenerator() {
  auto G = doc::findGeneratorByName("json");
  if (!G)
    return nullptr;
  return std::move(G.get());
}

class JSONGeneratorTest : public ClangDocContextTest {};

TEST_F(JSONGeneratorTest, emitRecordJSON) {
  RecordInfo I;
  I.Name = "Foo";
  I.IsTypeDef = false;
  Reference Ns[] = {
      Reference(EmptySID, "GlobalNamespace", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);
  I.Path = "GlobalNamespace";
  I.DefLoc = Location(1, 1, "main.cpp");
  I.TagType = TagTypeKind::Class;

  I.Template = TemplateInfo();
  TemplateParamInfo TParams[] = {TemplateParamInfo("class T")};
  I.Template->Params = llvm::ArrayRef(TParams);

  EnumInfo E;
  E.Name = "Color";
  E.Scoped = false;
  EnumValueInfo EV[] = {EnumValueInfo("RED", "0")};
  E.Members = llvm::ArrayRef(EV);
  InfoNode<EnumInfo> ENode(&E);
  I.Children.Enums.push_back(ENode);

  MemberTypeInfo M[] = {
      MemberTypeInfo(TypeInfo("int"), "X", AccessSpecifier::AS_protected)};
  I.Members = llvm::ArrayRef(M);

  BaseRecordInfo B(EmptySID, "F", "path/to/F", true, AccessSpecifier::AS_public,
                   true);
  FunctionInfo F;
  F.Name = "InheritedFunctionOne";
  InfoNode<FunctionInfo> FNode(&F);
  B.Children.Functions.push_back(FNode);
  MemberTypeInfo BM[] = {
      MemberTypeInfo(TypeInfo("int"), "N", AccessSpecifier::AS_public)};
  B.Members = llvm::ArrayRef(BM);

  BaseRecordInfo Bases[] = {std::move(B)};
  I.Bases = llvm::ArrayRef(Bases);

  // F is in the global namespace
  Reference Parents[] = {Reference(EmptySID, "F", InfoType::IT_record, "")};
  I.Parents = llvm::ArrayRef(Parents);
  Reference VParents[] = {Reference(EmptySID, "G", InfoType::IT_record,
                                    "path::to::G::G", "path/to/G")};
  I.VirtualParents = llvm::ArrayRef(VParents);

  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record,
                        "path::to::A::r::ChildStruct", "path/to/A/r");
  InfoNode<Reference> ChildStructNode(&ChildStruct);
  I.Children.Records.push_back(ChildStructNode);

  FunctionInfo F2;
  F2.Name = "OneFunction";
  InfoNode<FunctionInfo> F2Node(&F2);
  I.Children.Functions.push_back(F2Node);

  auto G = getJSONGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected = R"raw({
  "Bases": [
    {
      "Access": "public",
      "End": true,
      "HasMembers": true,
      "HasPublicMembers": true,
      "HasPublicMethods": true,
      "InfoType": "record",
      "IsParent": true,
      "IsTypedef": false,
      "IsVirtual": true,
      "MangledName": "",
      "Name": "F",
      "Path": "path/to/F",
      "PublicMembers": [
        {
          "IsStatic": false,
          "Name": "N",
          "Type": "int"
        }
      ],
      "PublicMethods": [
        {
          "InfoType": "function",
          "IsStatic": false,
          "Name": "InheritedFunctionOne",
          "ReturnType": {
            "IsBuiltIn": false,
            "IsTemplate": false,
            "Name": "",
            "QualName": "",
            "USR": "0000000000000000000000000000000000000000"
          }
        }
      ],
      "TagType": "struct"
    }
  ],
  "Enums": [
    {
      "End": true,
      "InfoType": "enum",
      "Members": [
        {
          "End": true,
          "Name": "RED",
          "Value": "0"
        }
      ],
      "Name": "Color",
      "Scoped": false
    }
  ],
  "HasEnums": true,
  "HasMembers": true,
  "HasParents": true,
  "HasProtectedMembers": true,
  "HasPublicMethods": true,
  "HasRecords": true,
  "HasVirtualParents": true,
  "InfoType": "record",
  "IsTypedef": false,
  "Location": {
    "Filename": "main.cpp",
    "LineNumber": 1
  },
  "MangledName": "",
  "Name": "Foo",
  "Namespace": [
    "GlobalNamespace"
  ],
  "Parents": [
    {
      "End": true,
      "Name": "F",
      "QualName": "",
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "Path": "GlobalNamespace",
  "ProtectedMembers": [
    {
      "IsStatic": false,
      "Name": "X",
      "Type": "int"
    }
  ],
  "PublicMethods": [
    {
      "InfoType": "function",
      "IsStatic": false,
      "Name": "OneFunction",
      "ReturnType": {
        "IsBuiltIn": false,
        "IsTemplate": false,
        "Name": "",
        "QualName": "",
        "USR": "0000000000000000000000000000000000000000"
      }
    }
  ],
  "Records": [
    {
      "End": true,
      "Name": "ChildStruct",
      "Path": "path/to/A/r",
      "QualName": "path::to::A::r::ChildStruct",
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "TagType": "class",
  "Template": {
    "Parameters": [
      {
        "End": true,
        "Param": "class T"
      }
    ],
    "VerticalDisplay": false
  },
  "VirtualParents": [
    {
      "End": true,
      "Name": "G",
      "Path": "path/to/G",
      "QualName": "path::to::G::G",
      "USR": "0000000000000000000000000000000000000000"
    }
  ]
})raw";
  EXPECT_EQ(Expected, Actual.str());
}

TEST_F(JSONGeneratorTest, emitNamespaceJSON) {
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Path = "path/to/A";
  Reference Ns[] = {Reference(EmptySID, "A", InfoType::IT_namespace)};
  I.Namespace = llvm::ArrayRef(Ns);

  Reference NewNamespace(EmptySID, "ChildNamespace", InfoType::IT_namespace,
                         "path::to::A::Namespace::ChildNamespace",
                         "path/to/A/Namespace");
  InfoNode<Reference> NewNamespaceNode(&NewNamespace);
  I.Children.Namespaces.push_back(NewNamespaceNode);

  Reference ChildStruct(EmptySID, "ChildStruct", InfoType::IT_record,
                        "path::to::A::Namespace::ChildStruct",
                        "path/to/A/Namespace");
  InfoNode<Reference> ChildStructNode(&ChildStruct);
  I.Children.Records.push_back(ChildStructNode);
  FunctionInfo F;
  F.Name = "OneFunction";
  F.Access = AccessSpecifier::AS_none;
  InfoNode<FunctionInfo> FNode(&F);
  I.Children.Functions.push_back(FNode);

  EnumInfo E;
  E.Name = "OneEnum";
  InfoNode<EnumInfo> ENode(&E);
  I.Children.Enums.push_back(ENode);

  auto G = getJSONGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, getClangDocContext());
  assert(!Err);
  std::string Expected = R"raw({
  "Enums": [
    {
      "End": true,
      "InfoType": "enum",
      "Name": "OneEnum",
      "Scoped": false
    }
  ],
  "Functions": [
    {
      "End": true,
      "InfoType": "function",
      "IsStatic": false,
      "Name": "OneFunction",
      "ReturnType": {
        "IsBuiltIn": false,
        "IsTemplate": false,
        "Name": "",
        "QualName": "",
        "USR": "0000000000000000000000000000000000000000"
      }
    }
  ],
  "HasEnums": true,
  "HasFunctions": true,
  "HasNamespaces": true,
  "HasRecords": true,
  "InfoType": "namespace",
  "Name": "Global Namespace",
  "Namespace": [
    "A"
  ],
  "Namespaces": [
    {
      "End": true,
      "Name": "ChildNamespace",
      "Path": "path/to/A/Namespace",
      "QualName": "path::to::A::Namespace::ChildNamespace",
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "Path": "path/to/A",
  "Records": [
    {
      "End": true,
      "Name": "ChildStruct",
      "Path": "path/to/A/Namespace",
      "QualName": "path::to::A::Namespace::ChildStruct",
      "USR": "0000000000000000000000000000000000000000"
    }
  ]
})raw";
  EXPECT_EQ(Expected, Actual.str());
}
} // namespace doc
} // namespace clang
