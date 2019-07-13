//===-- HTMLGenerator.cpp - HTML Generator ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "Representation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <string>

using namespace llvm;

namespace clang {
namespace doc {

namespace {

class HTMLTag {
public:
  // Any other tag can be added if required
  enum TagType {
    TAG_META,
    TAG_TITLE,
    TAG_DIV,
    TAG_H1,
    TAG_H2,
    TAG_H3,
    TAG_P,
    TAG_UL,
    TAG_LI,
    TAG_A,
  };

  HTMLTag() = default;
  constexpr HTMLTag(TagType Value) : Value(Value) {}

  operator TagType() const { return Value; }
  operator bool() = delete;

  bool IsSelfClosing() const;

  bool HasInlineChildren() const;

  llvm::SmallString<16> ToString() const;

private:
  TagType Value;
};

enum NodeType {
  NODE_TEXT,
  NODE_TAG,
};

struct HTMLNode {
  HTMLNode(NodeType Type) : Type(Type) {}
  virtual ~HTMLNode() = default;

  virtual void Render(llvm::raw_ostream &OS, int IndentationLevel) = 0;
  NodeType Type; // Type of node
};

struct TextNode : public HTMLNode {
  TextNode(const Twine &Text, bool Indented = true)
      : HTMLNode(NodeType::NODE_TEXT), Text(Text.str()), Indented(Indented) {}

  std::string Text; // Content of node
  bool Indented; // Indicates if an indentation must be rendered before the text
  void Render(llvm::raw_ostream &OS, int IndentationLevel) override;
};

struct TagNode : public HTMLNode {
  TagNode(HTMLTag Tag)
      : HTMLNode(NodeType::NODE_TAG), Tag(Tag),
        InlineChildren(Tag.HasInlineChildren()),
        SelfClosing(Tag.IsSelfClosing()) {}
  TagNode(HTMLTag Tag, const Twine &Text) : TagNode(Tag) {
    Children.emplace_back(
        llvm::make_unique<TextNode>(Text.str(), !InlineChildren));
  }

  HTMLTag Tag;         // Name of HTML Tag (p, div, h1)
  bool InlineChildren; // Indicates if children nodes are rendered in the same
                       // line as itself or if children must rendered in the
                       // next line and with additional indentation
  bool SelfClosing;    // Indicates if tag is self-closing
  std::vector<std::unique_ptr<HTMLNode>> Children; // List of child nodes
  llvm::StringMap<llvm::SmallString<16>>
      Attributes; // List of key-value attributes for tag

  void Render(llvm::raw_ostream &OS, int IndentationLevel) override;
};

constexpr const char *kDoctypeDecl = "<!DOCTYPE html>";

struct HTMLFile {
  std::vector<std::unique_ptr<HTMLNode>> Children; // List of child nodes
  void Render(llvm::raw_ostream &OS) {
    OS << kDoctypeDecl << "\n";
    for (const auto &C : Children) {
      C->Render(OS, 0);
      OS << "\n";
    }
  }
};

} // namespace

bool HTMLTag::IsSelfClosing() const {
  switch (Value) {
  case HTMLTag::TAG_META:
    return true;
  case HTMLTag::TAG_TITLE:
  case HTMLTag::TAG_DIV:
  case HTMLTag::TAG_H1:
  case HTMLTag::TAG_H2:
  case HTMLTag::TAG_H3:
  case HTMLTag::TAG_P:
  case HTMLTag::TAG_UL:
  case HTMLTag::TAG_LI:
  case HTMLTag::TAG_A:
    return false;
  }
  llvm_unreachable("Unhandled HTMLTag::TagType");
}

bool HTMLTag::HasInlineChildren() const {
  switch (Value) {
  case HTMLTag::TAG_META:
  case HTMLTag::TAG_TITLE:
  case HTMLTag::TAG_H1:
  case HTMLTag::TAG_H2:
  case HTMLTag::TAG_H3:
  case HTMLTag::TAG_LI:
  case HTMLTag::TAG_A:
    return true;
  case HTMLTag::TAG_DIV:
  case HTMLTag::TAG_P:
  case HTMLTag::TAG_UL:
    return false;
  }
  llvm_unreachable("Unhandled HTMLTag::TagType");
}

llvm::SmallString<16> HTMLTag::ToString() const {
  switch (Value) {
  case HTMLTag::TAG_META:
    return llvm::SmallString<16>("meta");
  case HTMLTag::TAG_TITLE:
    return llvm::SmallString<16>("title");
  case HTMLTag::TAG_DIV:
    return llvm::SmallString<16>("div");
  case HTMLTag::TAG_H1:
    return llvm::SmallString<16>("h1");
  case HTMLTag::TAG_H2:
    return llvm::SmallString<16>("h2");
  case HTMLTag::TAG_H3:
    return llvm::SmallString<16>("h3");
  case HTMLTag::TAG_P:
    return llvm::SmallString<16>("p");
  case HTMLTag::TAG_UL:
    return llvm::SmallString<16>("ul");
  case HTMLTag::TAG_LI:
    return llvm::SmallString<16>("li");
  case HTMLTag::TAG_A:
    return llvm::SmallString<16>("a");
  }
  llvm_unreachable("Unhandled HTMLTag::TagType");
}

void TextNode::Render(llvm::raw_ostream &OS, int IndentationLevel) {
  if (Indented)
    OS.indent(IndentationLevel * 2);
  OS << Text;
}

void TagNode::Render(llvm::raw_ostream &OS, int IndentationLevel) {
  OS.indent(IndentationLevel * 2);
  OS << "<" << Tag.ToString();
  for (const auto &A : Attributes)
    OS << " " << A.getKey() << "=\"" << A.getValue() << "\"";
  if (SelfClosing) {
    OS << "/>";
    return;
  }
  OS << ">";
  if (!InlineChildren)
    OS << "\n";
  bool NewLineRendered = true;
  for (const auto &C : Children) {
    int ChildrenIndentation =
        InlineChildren || !NewLineRendered ? 0 : IndentationLevel + 1;
    C->Render(OS, ChildrenIndentation);
    if (!InlineChildren && (C == Children.back() ||
                            (C->Type != NodeType::NODE_TEXT ||
                             (&C + 1)->get()->Type != NodeType::NODE_TEXT))) {
      OS << "\n";
      NewLineRendered = true;
    } else
      NewLineRendered = false;
  }
  if (!InlineChildren)
    OS.indent(IndentationLevel * 2);
  OS << "</" << Tag.ToString() << ">";
}

template <typename Derived, typename Base,
          typename = std::enable_if<std::is_base_of<Derived, Base>::value>>
static void AppendVector(std::vector<Derived> &&New,
                         std::vector<Base> &Original) {
  std::move(New.begin(), New.end(), std::back_inserter(Original));
}

// Compute the relative path that names the file path relative to the given
// directory.
static SmallString<128> computeRelativePath(StringRef FilePath,
                                            StringRef Directory) {
  StringRef Path = FilePath;
  while (!Path.empty()) {
    if (Directory == Path)
      return FilePath.substr(Path.size());
    Path = llvm::sys::path::parent_path(Path);
  }

  StringRef Dir = Directory;
  SmallString<128> Result;
  while (!Dir.empty()) {
    if (Dir == FilePath)
      break;
    Dir = llvm::sys::path::parent_path(Dir);
    llvm::sys::path::append(Result, "..");
  }
  llvm::sys::path::append(Result, FilePath.substr(Dir.size()));
  return Result;
}

// HTML generation

static std::unique_ptr<TagNode> genLink(const Twine &Text, const Twine &Link) {
  auto LinkNode = llvm::make_unique<TagNode>(HTMLTag::TAG_A, Text);
  LinkNode->Attributes.try_emplace("href", Link.str());
  return LinkNode;
}

static std::unique_ptr<HTMLNode> genTypeReference(const Reference &Type,
                                                  StringRef CurrentDirectory) {
  if (Type.Path.empty())
    return llvm::make_unique<TextNode>(Type.Name);
  llvm::SmallString<128> Path =
      computeRelativePath(Type.Path, CurrentDirectory);
  llvm::sys::path::append(Path, Type.Name + ".html");
  return genLink(Type.Name, Path);
}

static std::vector<std::unique_ptr<HTMLNode>>
genReferenceList(const llvm::SmallVectorImpl<Reference> &Refs,
                 const StringRef &CurrentDirectory) {
  std::vector<std::unique_ptr<HTMLNode>> Out;
  for (const auto &R : Refs) {
    if (&R != Refs.begin())
      Out.emplace_back(llvm::make_unique<TextNode>(", "));
    Out.emplace_back(genTypeReference(R, CurrentDirectory));
  }
  return Out;
}

static std::vector<std::unique_ptr<TagNode>> genHTML(const EnumInfo &I);
static std::vector<std::unique_ptr<TagNode>> genHTML(const FunctionInfo &I,
                                                     StringRef ParentInfoDir);

static std::vector<std::unique_ptr<TagNode>>
genEnumsBlock(const std::vector<EnumInfo> &Enums) {
  if (Enums.empty())
    return {};

  std::vector<std::unique_ptr<TagNode>> Out;
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H2, "Enums"));
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_DIV));
  auto &DivBody = Out.back();
  for (const auto &E : Enums) {
    std::vector<std::unique_ptr<TagNode>> Nodes = genHTML(E);
    AppendVector(std::move(Nodes), DivBody->Children);
  }
  return Out;
}

static std::unique_ptr<TagNode>
genEnumMembersBlock(const llvm::SmallVector<SmallString<16>, 4> &Members) {
  if (Members.empty())
    return nullptr;

  auto List = llvm::make_unique<TagNode>(HTMLTag::TAG_UL);
  for (const auto &M : Members)
    List->Children.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_LI, M));
  return List;
}

static std::vector<std::unique_ptr<TagNode>>
genFunctionsBlock(const std::vector<FunctionInfo> &Functions,
                  StringRef ParentInfoDir) {
  if (Functions.empty())
    return {};

  std::vector<std::unique_ptr<TagNode>> Out;
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H2, "Functions"));
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_DIV));
  auto &DivBody = Out.back();
  for (const auto &F : Functions) {
    std::vector<std::unique_ptr<TagNode>> Nodes = genHTML(F, ParentInfoDir);
    AppendVector(std::move(Nodes), DivBody->Children);
  }
  return Out;
}

static std::vector<std::unique_ptr<TagNode>>
genRecordMembersBlock(const llvm::SmallVector<MemberTypeInfo, 4> &Members,
                      StringRef ParentInfoDir) {
  if (Members.empty())
    return {};

  std::vector<std::unique_ptr<TagNode>> Out;
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H2, "Members"));
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_UL));
  auto &ULBody = Out.back();
  for (const auto &M : Members) {
    std::string Access = getAccess(M.Access);
    if (Access != "")
      Access = Access + " ";
    auto LIBody = llvm::make_unique<TagNode>(HTMLTag::TAG_LI);
    LIBody->Children.emplace_back(llvm::make_unique<TextNode>(Access));
    LIBody->Children.emplace_back(genTypeReference(M.Type, ParentInfoDir));
    LIBody->Children.emplace_back(llvm::make_unique<TextNode>(" " + M.Name));
    ULBody->Children.emplace_back(std::move(LIBody));
  }
  return Out;
}

static std::vector<std::unique_ptr<TagNode>>
genReferencesBlock(const std::vector<Reference> &References,
                   llvm::StringRef Title) {
  if (References.empty())
    return {};

  std::vector<std::unique_ptr<TagNode>> Out;
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H2, Title));
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_UL));
  auto &ULBody = Out.back();
  for (const auto &R : References)
    ULBody->Children.emplace_back(
        llvm::make_unique<TagNode>(HTMLTag::TAG_LI, R.Name));
  return Out;
}

static std::unique_ptr<TagNode> writeFileDefinition(const Location &L) {
  return llvm::make_unique<TagNode>(
      HTMLTag::TAG_P,
      "Defined at line " + std::to_string(L.LineNumber) + " of " + L.Filename);
}

static std::unique_ptr<HTMLNode> genHTML(const CommentInfo &I) {
  if (I.Kind == "FullComment") {
    auto FullComment = llvm::make_unique<TagNode>(HTMLTag::TAG_DIV);
    for (const auto &Child : I.Children) {
      std::unique_ptr<HTMLNode> Node = genHTML(*Child);
      if (Node)
        FullComment->Children.emplace_back(std::move(Node));
    }
    return std::move(FullComment);
  } else if (I.Kind == "ParagraphComment") {
    auto ParagraphComment = llvm::make_unique<TagNode>(HTMLTag::TAG_P);
    for (const auto &Child : I.Children) {
      std::unique_ptr<HTMLNode> Node = genHTML(*Child);
      if (Node)
        ParagraphComment->Children.emplace_back(std::move(Node));
    }
    if (ParagraphComment->Children.empty())
      return nullptr;
    return std::move(ParagraphComment);
  } else if (I.Kind == "TextComment") {
    if (I.Text == "")
      return nullptr;
    return llvm::make_unique<TextNode>(I.Text, true);
  }
  return nullptr;
}

static std::unique_ptr<TagNode> genHTML(const std::vector<CommentInfo> &C) {
  auto CommentBlock = llvm::make_unique<TagNode>(HTMLTag::TAG_DIV);
  for (const auto &Child : C) {
    if (std::unique_ptr<HTMLNode> Node = genHTML(Child))
      CommentBlock->Children.emplace_back(std::move(Node));
  }
  return CommentBlock;
}

static std::vector<std::unique_ptr<TagNode>> genHTML(const EnumInfo &I) {
  std::vector<std::unique_ptr<TagNode>> Out;
  std::string EnumType;
  if (I.Scoped)
    EnumType = "enum class ";
  else
    EnumType = "enum ";

  Out.emplace_back(
      llvm::make_unique<TagNode>(HTMLTag::TAG_H3, EnumType + I.Name));

  std::unique_ptr<TagNode> Node = genEnumMembersBlock(I.Members);
  if (Node)
    Out.emplace_back(std::move(Node));

  if (I.DefLoc)
    Out.emplace_back(writeFileDefinition(I.DefLoc.getValue()));

  std::string Description;
  if (!I.Description.empty())
    Out.emplace_back(genHTML(I.Description));

  return Out;
}

static std::vector<std::unique_ptr<TagNode>> genHTML(const FunctionInfo &I,
                                                     StringRef ParentInfoDir) {
  std::vector<std::unique_ptr<TagNode>> Out;
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H3, I.Name));

  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_P));
  auto &FunctionHeader = Out.back();

  std::string Access = getAccess(I.Access);
  if (Access != "")
    FunctionHeader->Children.emplace_back(
        llvm::make_unique<TextNode>(Access + " "));
  if (I.ReturnType.Type.Name != "") {
    FunctionHeader->Children.emplace_back(
        genTypeReference(I.ReturnType.Type, ParentInfoDir));
    FunctionHeader->Children.emplace_back(llvm::make_unique<TextNode>(" "));
  }
  FunctionHeader->Children.emplace_back(
      llvm::make_unique<TextNode>(I.Name + "("));

  for (const auto &P : I.Params) {
    if (&P != I.Params.begin())
      FunctionHeader->Children.emplace_back(llvm::make_unique<TextNode>(", "));
    FunctionHeader->Children.emplace_back(
        genTypeReference(P.Type, ParentInfoDir));
    FunctionHeader->Children.emplace_back(
        llvm::make_unique<TextNode>(" " + P.Name));
  }
  FunctionHeader->Children.emplace_back(llvm::make_unique<TextNode>(")"));

  if (I.DefLoc)
    Out.emplace_back(writeFileDefinition(I.DefLoc.getValue()));

  std::string Description;
  if (!I.Description.empty())
    Out.emplace_back(genHTML(I.Description));

  return Out;
}

static std::vector<std::unique_ptr<TagNode>> genHTML(const NamespaceInfo &I,
                                                     std::string &InfoTitle) {
  std::vector<std::unique_ptr<TagNode>> Out;
  if (I.Name.str() == "")
    InfoTitle = "Global Namespace";
  else
    InfoTitle = ("namespace " + I.Name).str();

  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H1, InfoTitle));

  std::string Description;
  if (!I.Description.empty())
    Out.emplace_back(genHTML(I.Description));

  std::vector<std::unique_ptr<TagNode>> ChildNamespaces =
      genReferencesBlock(I.ChildNamespaces, "Namespaces");
  AppendVector(std::move(ChildNamespaces), Out);
  std::vector<std::unique_ptr<TagNode>> ChildRecords =
      genReferencesBlock(I.ChildRecords, "Records");
  AppendVector(std::move(ChildRecords), Out);

  std::vector<std::unique_ptr<TagNode>> ChildFunctions =
      genFunctionsBlock(I.ChildFunctions, I.Path);
  AppendVector(std::move(ChildFunctions), Out);
  std::vector<std::unique_ptr<TagNode>> ChildEnums =
      genEnumsBlock(I.ChildEnums);
  AppendVector(std::move(ChildEnums), Out);

  return Out;
}

static std::vector<std::unique_ptr<TagNode>> genHTML(const RecordInfo &I,
                                                     std::string &InfoTitle) {
  std::vector<std::unique_ptr<TagNode>> Out;
  InfoTitle = (getTagType(I.TagType) + " " + I.Name).str();
  Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_H1, InfoTitle));

  if (I.DefLoc)
    Out.emplace_back(writeFileDefinition(I.DefLoc.getValue()));

  std::string Description;
  if (!I.Description.empty())
    Out.emplace_back(genHTML(I.Description));

  std::vector<std::unique_ptr<HTMLNode>> Parents =
      genReferenceList(I.Parents, I.Path);
  std::vector<std::unique_ptr<HTMLNode>> VParents =
      genReferenceList(I.VirtualParents, I.Path);
  if (!Parents.empty() || !VParents.empty()) {
    Out.emplace_back(llvm::make_unique<TagNode>(HTMLTag::TAG_P));
    auto &PBody = Out.back();
    PBody->Children.emplace_back(llvm::make_unique<TextNode>("Inherits from "));
    if (Parents.empty())
      AppendVector(std::move(VParents), PBody->Children);
    else if (VParents.empty())
      AppendVector(std::move(Parents), PBody->Children);
    else {
      AppendVector(std::move(Parents), PBody->Children);
      PBody->Children.emplace_back(llvm::make_unique<TextNode>(", "));
      AppendVector(std::move(VParents), PBody->Children);
    }
  }

  std::vector<std::unique_ptr<TagNode>> Members =
      genRecordMembersBlock(I.Members, I.Path);
  AppendVector(std::move(Members), Out);
  std::vector<std::unique_ptr<TagNode>> ChildRecords =
      genReferencesBlock(I.ChildRecords, "Records");
  AppendVector(std::move(ChildRecords), Out);

  std::vector<std::unique_ptr<TagNode>> ChildFunctions =
      genFunctionsBlock(I.ChildFunctions, I.Path);
  AppendVector(std::move(ChildFunctions), Out);
  std::vector<std::unique_ptr<TagNode>> ChildEnums =
      genEnumsBlock(I.ChildEnums);
  AppendVector(std::move(ChildEnums), Out);

  return Out;
}

/// Generator for HTML documentation.
class HTMLGenerator : public Generator {
public:
  static const char *Format;

  llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS) override;
};

const char *HTMLGenerator::Format = "html";

llvm::Error HTMLGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS) {
  HTMLFile F;

  auto MetaNode = llvm::make_unique<TagNode>(HTMLTag::TAG_META);
  MetaNode->Attributes.try_emplace("charset", "utf-8");
  F.Children.emplace_back(std::move(MetaNode));

  std::string InfoTitle;
  Info CastedInfo;
  auto MainContentNode = llvm::make_unique<TagNode>(HTMLTag::TAG_DIV);
  switch (I->IT) {
  case InfoType::IT_namespace: {
    std::vector<std::unique_ptr<TagNode>> Nodes =
        genHTML(*static_cast<clang::doc::NamespaceInfo *>(I), InfoTitle);
    AppendVector(std::move(Nodes), MainContentNode->Children);
    break;
  }
  case InfoType::IT_record: {
    std::vector<std::unique_ptr<TagNode>> Nodes =
        genHTML(*static_cast<clang::doc::RecordInfo *>(I), InfoTitle);
    AppendVector(std::move(Nodes), MainContentNode->Children);
    break;
  }
  case InfoType::IT_enum: {
    std::vector<std::unique_ptr<TagNode>> Nodes =
        genHTML(*static_cast<clang::doc::EnumInfo *>(I));
    AppendVector(std::move(Nodes), MainContentNode->Children);
    break;
  }
  case InfoType::IT_function: {
    std::vector<std::unique_ptr<TagNode>> Nodes =
        genHTML(*static_cast<clang::doc::FunctionInfo *>(I), "");
    AppendVector(std::move(Nodes), MainContentNode->Children);
    break;
  }
  case InfoType::IT_default:
    return llvm::make_error<llvm::StringError>("Unexpected info type.\n",
                                               llvm::inconvertibleErrorCode());
  }

  F.Children.emplace_back(
      llvm::make_unique<TagNode>(HTMLTag::TAG_TITLE, InfoTitle));
  F.Children.emplace_back(std::move(MainContentNode));
  F.Render(OS);

  return llvm::Error::success();
}

static GeneratorRegistry::Add<HTMLGenerator> HTML(HTMLGenerator::Format,
                                                  "Generator for HTML output.");

// This anchor is used to force the linker to link in the generated object
// file and thus register the generator.
volatile int HTMLGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
