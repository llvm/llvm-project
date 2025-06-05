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
  I.FullName = "Foo";
  I.IsTypeDef = false;
  I.Namespace.emplace_back(EmptySID, "GlobalNamespace", InfoType::IT_namespace);
  I.Path = "GlobalNamespace";
  I.DefLoc = Location(1, 1, "main.cpp");
  I.TagType = TagTypeKind::Class;

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
      "FullName": "F",
      "IsParent": true,
      "IsTypedef": false,
      "IsVirtual": true,
      "Name": "F",
      "Path": "path/to/F",
      "PublicFunctions": [
        {
          "IsStatic": false,
          "Name": "InheritedFunctionOne",
          "ReturnType": {
            "ID": "0000000000000000000000000000000000000000",
            "IsBuiltIn": false,
            "IsTemplate": false,
            "Name": "",
            "QualName": ""
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
      "Members": [
        {
          "Name": "RED",
          "Value": "0"
        }
      ],
      "Name": "Color",
      "Scoped": false,
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "FullName": "Foo",
  "IsTypedef": false,
  "Location": {
    "Filename": "main.cpp",
    "LineNumber": 1
  },
  "Name": "Foo",
  "Namespace": [
    "GlobalNamespace"
  ],
  "Parents": [
    {
      "ID": "0000000000000000000000000000000000000000",
      "Link": "F.json",
      "Name": "F",
      "QualName": ""
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
      "IsStatic": false,
      "Name": "OneFunction",
      "ReturnType": {
        "ID": "0000000000000000000000000000000000000000",
        "IsBuiltIn": false,
        "IsTemplate": false,
        "Name": "",
        "QualName": ""
      },
      "USR": "0000000000000000000000000000000000000000"
    }
  ],
  "Records": [
    {
      "ID": "0000000000000000000000000000000000000000",
      "Link": "ChildStruct.json",
      "Name": "ChildStruct",
      "QualName": "path::to::A::r::ChildStruct"
    }
  ],
  "TagType": "class",
  "USR": "0000000000000000000000000000000000000000",
  "VirtualParents": [
    {
      "ID": "0000000000000000000000000000000000000000",
      "Link": "G.json",
      "Name": "G",
      "QualName": "path::to::G::G"
    }
  ]
})raw";
  EXPECT_EQ(Expected, Actual.str());
}
} // namespace doc
} // namespace clang
