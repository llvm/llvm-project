//===- unittest/Tooling/CommentHandlerTest.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {

struct Comment {
  Comment(const std::string &Message, unsigned Line, unsigned Col)
    : Message(Message), Line(Line), Col(Col) { }

  std::string Message;
  unsigned Line, Col;
};

class CommentVerifier;
typedef std::vector<Comment> CommentList;

class CommentHandlerVisitor : public TestVisitor<CommentHandlerVisitor>,
                              public CommentHandler {
  typedef TestVisitor<CommentHandlerVisitor> base;

public:
  CommentHandlerVisitor() : base(), PP(nullptr), Verified(false) {}

  ~CommentHandlerVisitor() override {
    EXPECT_TRUE(Verified) << "CommentVerifier not accessed";
  }

  bool HandleComment(Preprocessor &PP, SourceRange Loc) override {
    assert(&PP == this->PP && "Preprocessor changed!");

    SourceLocation Start = Loc.getBegin();
    SourceManager &SM = PP.getSourceManager();
    std::string C(SM.getCharacterData(Start),
                  SM.getCharacterData(Loc.getEnd()));

    bool Invalid;
    unsigned CLine = SM.getSpellingLineNumber(Start, &Invalid);
    EXPECT_TRUE(!Invalid) << "Invalid line number on comment " << C;

    unsigned CCol = SM.getSpellingColumnNumber(Start, &Invalid);
    EXPECT_TRUE(!Invalid) << "Invalid column number on comment " << C;

    Comments.push_back(Comment(C, CLine, CCol));
    return false;
  }

  CommentVerifier GetVerifier();

protected:
  ASTFrontendAction *CreateTestAction() override {
    return new CommentHandlerAction(this);
  }

private:
  Preprocessor *PP;
  CommentList Comments;
  bool Verified;

  class CommentHandlerAction : public base::TestAction {
  public:
    CommentHandlerAction(CommentHandlerVisitor *Visitor)
        : TestAction(Visitor) { }

    bool BeginSourceFileAction(CompilerInstance &CI) override {
      CommentHandlerVisitor *V =
          static_cast<CommentHandlerVisitor*>(this->Visitor);
      V->PP = &CI.getPreprocessor();
      V->PP->addCommentHandler(V);
      return true;
    }

    void EndSourceFileAction() override {
      CommentHandlerVisitor *V =
          static_cast<CommentHandlerVisitor*>(this->Visitor);
      V->PP->removeCommentHandler(V);
    }
  };
};

class CommentVerifier {
  CommentList::const_iterator Current;
  CommentList::const_iterator End;
  Preprocessor *PP;

public:
  CommentVerifier(const CommentList &Comments, Preprocessor *PP)
      : Current(Comments.begin()), End(Comments.end()), PP(PP)
    { }

  CommentVerifier(CommentVerifier &&C) : Current(C.Current), End(C.End), PP(C.PP) {
    C.Current = C.End;
  }

  ~CommentVerifier() {
    if (Current != End) {
      EXPECT_TRUE(Current == End) << "Unexpected comment \""
        << Current->Message << "\" at line " << Current->Line << ", column "
        << Current->Col;
    }
  }

  void Match(const char *Message, unsigned Line, unsigned Col) {
    EXPECT_TRUE(Current != End) << "Comment " << Message << " not found";
    if (Current == End) return;

    const Comment &C = *Current;
    EXPECT_TRUE(C.Message == Message && C.Line == Line && C.Col == Col)
      <<   "Expected comment \"" << Message
      << "\" at line " << Line   << ", column " << Col
      << "\nActual comment   \"" << C.Message
      << "\" at line " << C.Line << ", column " << C.Col;

    ++Current;
  }
};

CommentVerifier CommentHandlerVisitor::GetVerifier() {
  Verified = true;
  return CommentVerifier(Comments, PP);
}


TEST(CommentHandlerTest, BasicTest1) {
  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver("class X {}; int main() { return 0; }"));
  CommentVerifier Verifier = Visitor.GetVerifier();
}

TEST(CommentHandlerTest, BasicTest2) {
  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver(
        "class X {}; int main() { /* comment */ return 0; }"));
  CommentVerifier Verifier = Visitor.GetVerifier();
  Verifier.Match("/* comment */", 1, 26);
}

TEST(CommentHandlerTest, BasicTest3) {
  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver(
        "class X {}; // comment 1\n"
        "int main() {\n"
        "  // comment 2\n"
        "  return 0;\n"
        "}"));
  CommentVerifier Verifier = Visitor.GetVerifier();
  Verifier.Match("// comment 1", 1, 13);
  Verifier.Match("// comment 2", 3, 3);
}

TEST(CommentHandlerTest, IfBlock1) {
  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver(
        "#if 0\n"
        "// ignored comment\n"
        "#endif\n"
        "// visible comment\n"));
  CommentVerifier Verifier = Visitor.GetVerifier();
  Verifier.Match("// visible comment", 4, 1);
}

TEST(CommentHandlerTest, IfBlock2) {
  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver(
        "#define TEST        // visible_1\n"
        "#ifndef TEST        // visible_2\n"
        "                    // ignored_3\n"
        "# ifdef UNDEFINED   // ignored_4\n"
        "# endif             // ignored_5\n"
        "#elif defined(TEST) // visible_6\n"
        "# if 1              // visible_7\n"
        "                    // visible_8\n"
        "# else              // visible_9\n"
        "                    // ignored_10\n"
        "#  ifndef TEST      // ignored_11\n"
        "#  endif            // ignored_12\n"
        "# endif             // visible_13\n"
        "#endif              // visible_14\n"));

  CommentVerifier Verifier = Visitor.GetVerifier();
  Verifier.Match("// visible_1", 1, 21);
  Verifier.Match("// visible_2", 2, 21);
  Verifier.Match("// visible_6", 6, 21);
  Verifier.Match("// visible_7", 7, 21);
  Verifier.Match("// visible_8", 8, 21);
  Verifier.Match("// visible_9", 9, 21);
  Verifier.Match("// visible_13", 13, 21);
  Verifier.Match("// visible_14", 14, 21);
}

TEST(CommentHandlerTest, IfBlock3) {
  const char *Source =
        "/* commented out ...\n"
        "#if 0\n"
        "// enclosed\n"
        "#endif */";

  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver(Source));
  CommentVerifier Verifier = Visitor.GetVerifier();
  Verifier.Match(Source, 1, 1);
}

TEST(CommentHandlerTest, PPDirectives) {
  CommentHandlerVisitor Visitor;
  EXPECT_TRUE(Visitor.runOver(
        "#warning Y   // ignored_1\n" // #warning takes whole line as message
        "#undef MACRO // visible_2\n"
        "#line 1      // visible_3\n"));

  CommentVerifier Verifier = Visitor.GetVerifier();
  Verifier.Match("// visible_2", 2, 14);
  Verifier.Match("// visible_3", 3, 14);
}

} // end namespace clang
