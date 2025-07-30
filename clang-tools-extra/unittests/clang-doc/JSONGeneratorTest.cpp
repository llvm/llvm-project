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

TEST(JSONGeneratorTest, emitRecordJSON) {
  RecordInfo I;
  I.Name = "Foo";
  // FIXME: FullName is not emitted correctly.
  I.FullName = "";
  I.IsTypeDef = false;
  I.Namespace.emplace_back(EmptySID, "GlobalNamespace", InfoType::IT_namespace);
  I.Path = "GlobalNamespace";
  I.DefLoc = Location(1, 1, "main.cpp");
  I.TagType = TagTypeKind::Class;

  I.Template = TemplateInfo();
  I.Template->Params.emplace_back("class T");

  I.Children.Enums.emplace_back();
  I.Children.Enums.back().Name = "Color";
  I.Children.Enums.back().Scoped = false;
  I.Children.Enums.back().Members.emplace_back();
  I.Children.Enums.back().Members.back().Name = "RED";
  I.Children.Enums.back().Members.back().Value = "0";

  I.Members.emplace_back(TypeInfo("int"), "X", AccessSpecifier::AS_protected);

  I.Bases.emplace_back(EmptySID, "F", "path/to/F", true,
                       AccessSpecifier::AS_public, true);
  I.Bases.back().Children.Functions.emplace_back();
  I.Bases.back().Children.Functions.back().Name = "InheritedFunctionOne";
  I.Bases.back().Members.emplace_back(TypeInfo("int"), "N",
                                      AccessSpecifier::AS_public);

  // F is in the global namespace
  I.Parents.emplace_back(EmptySID, "F", InfoType::IT_record, "");
  I.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record,
                                "path::to::G::G", "path/to/G");

  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record,
                                  "path::to::A::r::ChildStruct", "path/to/A/r");
  I.Children.Functions.emplace_back();
  I.Children.Functions.back().Name = "OneFunction";

  auto G = getJSONGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected = R"raw({
  "Bases": [
    {
      "Access": "public",
      "End": true,
      "FullName": "",
      "HasPublicFunctions": true,
      "HasPublicMembers": true,
      "InfoType": "record",
      "IsParent": true,
      "IsTypedef": false,
      "IsVirtual": true,
      "MangledName": "",
      "Name": "F",
      "Path": "path/to/F",
      "PublicFunctions": [
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
          },
          "USR": "0000000000000000000000000000000000000000"
        }
      ],
      "PublicMembers": [
        {
          "Name": "N",
          "Type": "int"
        }
      ],
      "TagType": "struct",
      "USR": "0000000000000000000000000000000000000000"
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
      "Scoped": false,
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "FullName": "",
  "HasEnums": true,
  "HasPublicFunctions": true,
  "HasRecords": true,
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
      "Path": "",
      "QualName": "",
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "Path": "GlobalNamespace",
  "ProtectedMembers": [
    {
      "Name": "X",
      "Type": "int"
    }
  ],
  "PublicFunctions": [
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
      },
      "USR": "0000000000000000000000000000000000000000"
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
      "class T"
    ]
  },
  "USR": "0000000000000000000000000000000000000000",
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

TEST(JSONGeneratorTest, emitNamespaceJSON) {
  NamespaceInfo I;
  I.Name = "Namespace";
  I.Path = "path/to/A";
  I.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  I.Children.Namespaces.emplace_back(
      EmptySID, "ChildNamespace", InfoType::IT_namespace,
      "path::to::A::Namespace::ChildNamespace", "path/to/A/Namespace");
  I.Children.Records.emplace_back(EmptySID, "ChildStruct", InfoType::IT_record,
                                  "path::to::A::Namespace::ChildStruct",
                                  "path/to/A/Namespace");
  I.Children.Functions.emplace_back();
  I.Children.Functions.back().Name = "OneFunction";
  I.Children.Functions.back().Access = AccessSpecifier::AS_none;
  I.Children.Enums.emplace_back();
  I.Children.Enums.back().Name = "OneEnum";

  auto G = getJSONGenerator();
  assert(G);
  std::string Buffer;
  llvm::raw_string_ostream Actual(Buffer);
  auto Err = G->generateDocForInfo(&I, Actual, ClangDocContext());
  assert(!Err);
  std::string Expected = R"raw({
  "Enums": [
    {
      "End": true,
      "InfoType": "enum",
      "Name": "OneEnum",
      "Scoped": false,
      "USR": "0000000000000000000000000000000000000000"
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
      },
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "HasEnums": true,
  "HasRecords": true,
  "InfoType": "namespace",
  "Name": "Namespace",
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
  ],
  "USR": "0000000000000000000000000000000000000000"
})raw";
  EXPECT_EQ(Expected, Actual.str());
}
} // namespace doc
} // namespace clang
