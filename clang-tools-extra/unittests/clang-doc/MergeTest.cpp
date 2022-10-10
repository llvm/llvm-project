//===-- clang-doc/MergeTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocTest.h"
#include "Representation.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

TEST(MergeTest, mergeNamespaceInfos) {
  NamespaceInfo One;
  One.Name = "Namespace";
  One.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  One.ChildNamespaces.emplace_back(NonEmptySID, "ChildNamespace",
                                   InfoType::IT_namespace);
  One.ChildRecords.emplace_back(NonEmptySID, "ChildStruct",
                                InfoType::IT_record);
  One.ChildFunctions.emplace_back();
  One.ChildFunctions.back().Name = "OneFunction";
  One.ChildFunctions.back().USR = NonEmptySID;
  One.ChildEnums.emplace_back();
  One.ChildEnums.back().Name = "OneEnum";
  One.ChildEnums.back().USR = NonEmptySID;

  NamespaceInfo Two;
  Two.Name = "Namespace";
  Two.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Two.ChildNamespaces.emplace_back(EmptySID, "OtherChildNamespace",
                                   InfoType::IT_namespace);
  Two.ChildRecords.emplace_back(EmptySID, "OtherChildStruct",
                                InfoType::IT_record);
  Two.ChildFunctions.emplace_back();
  Two.ChildFunctions.back().Name = "TwoFunction";
  Two.ChildEnums.emplace_back();
  Two.ChildEnums.back().Name = "TwoEnum";

  std::vector<std::unique_ptr<Info>> Infos;
  Infos.emplace_back(std::make_unique<NamespaceInfo>(std::move(One)));
  Infos.emplace_back(std::make_unique<NamespaceInfo>(std::move(Two)));

  auto Expected = std::make_unique<NamespaceInfo>();
  Expected->Name = "Namespace";
  Expected->Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Expected->ChildNamespaces.emplace_back(NonEmptySID, "ChildNamespace",
                                         InfoType::IT_namespace);
  Expected->ChildRecords.emplace_back(NonEmptySID, "ChildStruct",
                                      InfoType::IT_record);
  Expected->ChildNamespaces.emplace_back(EmptySID, "OtherChildNamespace",
                                         InfoType::IT_namespace);
  Expected->ChildRecords.emplace_back(EmptySID, "OtherChildStruct",
                                      InfoType::IT_record);
  Expected->ChildFunctions.emplace_back();
  Expected->ChildFunctions.back().Name = "OneFunction";
  Expected->ChildFunctions.back().USR = NonEmptySID;
  Expected->ChildFunctions.emplace_back();
  Expected->ChildFunctions.back().Name = "TwoFunction";
  Expected->ChildEnums.emplace_back();
  Expected->ChildEnums.back().Name = "OneEnum";
  Expected->ChildEnums.back().USR = NonEmptySID;
  Expected->ChildEnums.emplace_back();
  Expected->ChildEnums.back().Name = "TwoEnum";

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckNamespaceInfo(InfoAsNamespace(Expected.get()),
                     InfoAsNamespace(Actual.get().get()));
}

TEST(MergeTest, mergeRecordInfos) {
  RecordInfo One;
  One.Name = "r";
  One.IsTypeDef = true;
  One.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  One.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});

  One.Members.emplace_back(TypeInfo("int"), "X", AccessSpecifier::AS_private);
  One.TagType = TagTypeKind::TTK_Class;
  One.Parents.emplace_back(EmptySID, "F", InfoType::IT_record);
  One.VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);

  One.Bases.emplace_back(EmptySID, "F", "path/to/F", true,
                         AccessSpecifier::AS_protected, true);
  One.ChildRecords.emplace_back(NonEmptySID, "SharedChildStruct",
                                InfoType::IT_record);
  One.ChildFunctions.emplace_back();
  One.ChildFunctions.back().Name = "OneFunction";
  One.ChildFunctions.back().USR = NonEmptySID;
  One.ChildEnums.emplace_back();
  One.ChildEnums.back().Name = "OneEnum";
  One.ChildEnums.back().USR = NonEmptySID;

  RecordInfo Two;
  Two.Name = "r";
  Two.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Two.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  Two.TagType = TagTypeKind::TTK_Class;

  Two.ChildRecords.emplace_back(NonEmptySID, "SharedChildStruct",
                                InfoType::IT_record, "path");
  Two.ChildFunctions.emplace_back();
  Two.ChildFunctions.back().Name = "TwoFunction";
  Two.ChildEnums.emplace_back();
  Two.ChildEnums.back().Name = "TwoEnum";

  std::vector<std::unique_ptr<Info>> Infos;
  Infos.emplace_back(std::make_unique<RecordInfo>(std::move(One)));
  Infos.emplace_back(std::make_unique<RecordInfo>(std::move(Two)));

  auto Expected = std::make_unique<RecordInfo>();
  Expected->Name = "r";
  Expected->IsTypeDef = true;
  Expected->Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Expected->DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  Expected->Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  Expected->Members.emplace_back(TypeInfo("int"), "X",
                                 AccessSpecifier::AS_private);
  Expected->TagType = TagTypeKind::TTK_Class;
  Expected->Parents.emplace_back(EmptySID, "F", InfoType::IT_record);
  Expected->VirtualParents.emplace_back(EmptySID, "G", InfoType::IT_record);
  Expected->Bases.emplace_back(EmptySID, "F", "path/to/F", true,
                               AccessSpecifier::AS_protected, true);

  Expected->ChildRecords.emplace_back(NonEmptySID, "SharedChildStruct",
                                      InfoType::IT_record, "path");
  Expected->ChildFunctions.emplace_back();
  Expected->ChildFunctions.back().Name = "OneFunction";
  Expected->ChildFunctions.back().USR = NonEmptySID;
  Expected->ChildFunctions.emplace_back();
  Expected->ChildFunctions.back().Name = "TwoFunction";
  Expected->ChildEnums.emplace_back();
  Expected->ChildEnums.back().Name = "OneEnum";
  Expected->ChildEnums.back().USR = NonEmptySID;
  Expected->ChildEnums.emplace_back();
  Expected->ChildEnums.back().Name = "TwoEnum";

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckRecordInfo(InfoAsRecord(Expected.get()),
                  InfoAsRecord(Actual.get().get()));
}

TEST(MergeTest, mergeFunctionInfos) {
  FunctionInfo One;
  One.Name = "f";
  One.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  One.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  One.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  One.IsMethod = true;
  One.Parent = Reference(EmptySID, "Parent", InfoType::IT_namespace);

  One.Description.emplace_back();
  auto OneFullComment = &One.Description.back();
  OneFullComment->Kind = "FullComment";
  auto OneParagraphComment = std::make_unique<CommentInfo>();
  OneParagraphComment->Kind = "ParagraphComment";
  auto OneTextComment = std::make_unique<CommentInfo>();
  OneTextComment->Kind = "TextComment";
  OneTextComment->Text = "This is a text comment.";
  OneParagraphComment->Children.push_back(std::move(OneTextComment));
  OneFullComment->Children.push_back(std::move(OneParagraphComment));

  FunctionInfo Two;
  Two.Name = "f";
  Two.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Two.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  Two.ReturnType = TypeInfo("void");
  Two.Params.emplace_back(TypeInfo("int"), "P");

  Two.Description.emplace_back();
  auto TwoFullComment = &Two.Description.back();
  TwoFullComment->Kind = "FullComment";
  auto TwoParagraphComment = std::make_unique<CommentInfo>();
  TwoParagraphComment->Kind = "ParagraphComment";
  auto TwoTextComment = std::make_unique<CommentInfo>();
  TwoTextComment->Kind = "TextComment";
  TwoTextComment->Text = "This is a text comment.";
  TwoParagraphComment->Children.push_back(std::move(TwoTextComment));
  TwoFullComment->Children.push_back(std::move(TwoParagraphComment));

  std::vector<std::unique_ptr<Info>> Infos;
  Infos.emplace_back(std::make_unique<FunctionInfo>(std::move(One)));
  Infos.emplace_back(std::make_unique<FunctionInfo>(std::move(Two)));

  auto Expected = std::make_unique<FunctionInfo>();
  Expected->Name = "f";
  Expected->Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Expected->DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  Expected->Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  Expected->ReturnType = TypeInfo("void");
  Expected->Params.emplace_back(TypeInfo("int"), "P");
  Expected->IsMethod = true;
  Expected->Parent = Reference(EmptySID, "Parent", InfoType::IT_namespace);

  Expected->Description.emplace_back();
  auto ExpectedFullComment = &Expected->Description.back();
  ExpectedFullComment->Kind = "FullComment";
  auto ExpectedParagraphComment = std::make_unique<CommentInfo>();
  ExpectedParagraphComment->Kind = "ParagraphComment";
  auto ExpectedTextComment = std::make_unique<CommentInfo>();
  ExpectedTextComment->Kind = "TextComment";
  ExpectedTextComment->Text = "This is a text comment.";
  ExpectedParagraphComment->Children.push_back(std::move(ExpectedTextComment));
  ExpectedFullComment->Children.push_back(std::move(ExpectedParagraphComment));

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckFunctionInfo(InfoAsFunction(Expected.get()),
                    InfoAsFunction(Actual.get().get()));
}

TEST(MergeTest, mergeEnumInfos) {
  EnumInfo One;
  One.Name = "e";
  One.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  One.DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  One.Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});

  One.Scoped = true;

  EnumInfo Two;
  Two.Name = "e";
  Two.Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Two.Loc.emplace_back(20, llvm::SmallString<16>{"test.cpp"});

  Two.Members.emplace_back("X");
  Two.Members.emplace_back("Y");

  std::vector<std::unique_ptr<Info>> Infos;
  Infos.emplace_back(std::make_unique<EnumInfo>(std::move(One)));
  Infos.emplace_back(std::make_unique<EnumInfo>(std::move(Two)));

  auto Expected = std::make_unique<EnumInfo>();
  Expected->Name = "e";
  Expected->Namespace.emplace_back(EmptySID, "A", InfoType::IT_namespace);

  Expected->DefLoc = Location(10, llvm::SmallString<16>{"test.cpp"});
  Expected->Loc.emplace_back(12, llvm::SmallString<16>{"test.cpp"});
  Expected->Loc.emplace_back(20, llvm::SmallString<16>{"test.cpp"});

  Expected->Members.emplace_back("X");
  Expected->Members.emplace_back("Y");
  Expected->Scoped = true;

  auto Actual = mergeInfos(Infos);
  assert(Actual);
  CheckEnumInfo(InfoAsEnum(Expected.get()), InfoAsEnum(Actual.get().get()));
}

} // namespace doc
} // namespace clang
