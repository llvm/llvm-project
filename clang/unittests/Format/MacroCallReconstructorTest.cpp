#include "../../lib/Format/Macros.h"
#include "../../lib/Format/UnwrappedLineParser.h"
#include "TestLexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <map>
#include <memory>
#include <vector>

namespace clang {
namespace format {
namespace {

using UnexpandedMap =
    llvm::DenseMap<FormatToken *, std::unique_ptr<UnwrappedLine>>;

// Keeps track of a sequence of macro expansions.
//
// The expanded tokens are accessible via getTokens(), while a map of macro call
// identifier token to unexpanded token stream is accessible via
// getUnexpanded().
class Expansion {
public:
  Expansion(TestLexer &Lex, MacroExpander &Macros) : Lex(Lex), Macros(Macros) {}

  // Appends the token stream obtained from expanding the macro Name given
  // the provided arguments, to be later retrieved with getTokens().
  // Returns the list of tokens making up the unexpanded macro call.
  TokenList expand(StringRef Name,
                   const SmallVector<SmallVector<FormatToken *, 8>, 1> &Args) {
    return expandInternal(Name, Args);
  }

  TokenList expand(StringRef Name) { return expandInternal(Name, {}); }

  TokenList expand(StringRef Name, const std::vector<std::string> &Args) {
    return expandInternal(Name, lexArgs(Args));
  }

  const UnexpandedMap &getUnexpanded() const { return Unexpanded; }

  const TokenList &getTokens() const { return Tokens; }

private:
  TokenList expandInternal(
      StringRef Name,
      const std::optional<SmallVector<SmallVector<FormatToken *, 8>, 1>>
          &Args) {
    auto *ID = Lex.id(Name);
    auto UnexpandedLine = std::make_unique<UnwrappedLine>();
    UnexpandedLine->Tokens.push_back(ID);
    if (Args && !Args->empty()) {
      UnexpandedLine->Tokens.push_back(Lex.id("("));
      for (auto I = Args->begin(), E = Args->end(); I != E; ++I) {
        if (I != Args->begin())
          UnexpandedLine->Tokens.push_back(Lex.id(","));
        UnexpandedLine->Tokens.insert(UnexpandedLine->Tokens.end(), I->begin(),
                                      I->end());
      }
      UnexpandedLine->Tokens.push_back(Lex.id(")"));
    }
    Unexpanded[ID] = std::move(UnexpandedLine);

    auto Expanded = uneof(Macros.expand(ID, Args));
    Tokens.append(Expanded.begin(), Expanded.end());

    TokenList UnexpandedTokens;
    for (const UnwrappedLineNode &Node : Unexpanded[ID]->Tokens)
      UnexpandedTokens.push_back(Node.Tok);
    return UnexpandedTokens;
  }

  SmallVector<TokenList, 1> lexArgs(const std::vector<std::string> &Args) {
    SmallVector<TokenList, 1> Result;
    for (const auto &Arg : Args)
      Result.push_back(uneof(Lex.lex(Arg)));
    return Result;
  }
  llvm::DenseMap<FormatToken *, std::unique_ptr<UnwrappedLine>> Unexpanded;
  SmallVector<FormatToken *, 8> Tokens;
  TestLexer &Lex;
  MacroExpander &Macros;
};

struct Chunk {
  Chunk(ArrayRef<FormatToken *> Tokens)
      : Tokens(Tokens.begin(), Tokens.end()) {}
  Chunk(ArrayRef<UnwrappedLine> Children)
      : Children(Children.begin(), Children.end()) {}
  SmallVector<UnwrappedLineNode, 1> Tokens;
  SmallVector<UnwrappedLine, 0> Children;
};

// Allows to produce chunks of a token list by typing the code of equal tokens.
//
// Created from a list of tokens, users call "consume" to get the next chunk
// of tokens, checking that they match the written code.
struct Matcher {
  Matcher(const TokenList &Tokens, TestLexer &Lex)
      : Tokens(Tokens), It(this->Tokens.begin()), Lex(Lex) {}

  bool tokenMatches(const FormatToken *Left, const FormatToken *Right) {
    if (Left->getType() == Right->getType() &&
        Left->TokenText == Right->TokenText) {
      return true;
    }
    llvm::dbgs() << Left->TokenText << " != " << Right->TokenText << "\n";
    return false;
  }

  Chunk consume(StringRef Tokens) {
    TokenList Result;
    for (const FormatToken *Token : uneof(Lex.lex(Tokens))) {
      (void)Token; // Fix unused variable warning when asserts are disabled.
      assert(tokenMatches(*It, Token));
      Result.push_back(*It);
      ++It;
    }
    return Chunk(Result);
  }

  TokenList Tokens;
  TokenList::iterator It;
  TestLexer &Lex;
};

UnexpandedMap mergeUnexpanded(const UnexpandedMap &M1,
                              const UnexpandedMap &M2) {
  UnexpandedMap Result;
  for (const auto &KV : M1)
    Result[KV.first] = std::make_unique<UnwrappedLine>(*KV.second);
  for (const auto &KV : M2)
    Result[KV.first] = std::make_unique<UnwrappedLine>(*KV.second);
  return Result;
}

class MacroCallReconstructorTest : public testing::Test {
public:
  MacroCallReconstructorTest() : Lex(Allocator, Buffers) {}

  std::unique_ptr<MacroExpander>
  createExpander(const std::vector<std::string> &MacroDefinitions) {
    return std::make_unique<MacroExpander>(MacroDefinitions,
                                           Lex.SourceMgr.get(), Lex.Style,
                                           Lex.Allocator, Lex.IdentTable);
  }

  UnwrappedLine line(ArrayRef<FormatToken *> Tokens, unsigned Level = 0) {
    UnwrappedLine Result;
    Result.Level = Level;
    for (FormatToken *Tok : Tokens)
      Result.Tokens.push_back(UnwrappedLineNode(Tok));
    return Result;
  }

  UnwrappedLine line(StringRef Text, unsigned Level = 0) {
    return line({lex(Text)}, Level);
  }

  UnwrappedLine line(ArrayRef<Chunk> Chunks, unsigned Level = 0) {
    UnwrappedLine Result;
    Result.Level = Level;
    for (const Chunk &Chunk : Chunks) {
      Result.Tokens.insert(Result.Tokens.end(), Chunk.Tokens.begin(),
                           Chunk.Tokens.end());
      assert(!Result.Tokens.empty());
      Result.Tokens.back().Children.append(Chunk.Children.begin(),
                                           Chunk.Children.end());
    }
    return Result;
  }

  TokenList lex(StringRef Text) { return uneof(Lex.lex(Text)); }

  Chunk tokens(StringRef Text) { return Chunk(lex(Text)); }

  Chunk children(ArrayRef<UnwrappedLine> Children) { return Chunk(Children); }

  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
  TestLexer Lex;
};

bool matchesTokens(const UnwrappedLine &L1, const UnwrappedLine &L2) {
  if (L1.Level != L2.Level)
    return false;
  if (L1.Tokens.size() != L2.Tokens.size())
    return false;
  for (auto L1It = L1.Tokens.begin(), L2It = L2.Tokens.begin();
       L1It != L1.Tokens.end(); ++L1It, ++L2It) {
    if (L1It->Tok != L2It->Tok)
      return false;
    if (L1It->Children.size() != L2It->Children.size())
      return false;
    for (auto L1ChildIt = L1It->Children.begin(),
              L2ChildIt = L2It->Children.begin();
         L1ChildIt != L1It->Children.end(); ++L1ChildIt, ++L2ChildIt) {
      if (!matchesTokens(*L1ChildIt, *L2ChildIt))
        return false;
    }
  }
  return true;
}
MATCHER_P(matchesLine, line, "") { return matchesTokens(arg, line); }

TEST_F(MacroCallReconstructorTest, Identifier) {
  auto Macros = createExpander({"X=x"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("X");

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Unexp.addLine(line(Exp.getTokens()));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(line(U.consume("X"))));
}

TEST_F(MacroCallReconstructorTest, NestedLineWithinCall) {
  auto Macros = createExpander({"C(a)=class X { a; };"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("C", {"void f()"});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line(E.consume("class X {")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("void f();")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("};")));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  EXPECT_THAT(std::move(Unexp).takeResult(),
              matchesLine(line(U.consume("C(void f())"))));
}

TEST_F(MacroCallReconstructorTest, MultipleLinesInNestedMultiParamsExpansion) {
  auto Macros = createExpander({"C(a, b)=a b", "B(a)={a}"});
  Expansion Exp1(Lex, *Macros);
  TokenList Call1 = Exp1.expand("B", {"b"});
  Expansion Exp2(Lex, *Macros);
  TokenList Call2 = Exp2.expand("C", {uneof(Lex.lex("a")), Exp1.getTokens()});

  UnexpandedMap Unexpanded =
      mergeUnexpanded(Exp1.getUnexpanded(), Exp2.getUnexpanded());
  MacroCallReconstructor Unexp(0, Unexpanded);
  Matcher E(Exp2.getTokens(), Lex);
  Unexp.addLine(line(E.consume("a")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("{")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("b")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("}")));
  EXPECT_TRUE(Unexp.finished());

  Matcher U1(Call1, Lex);
  auto Middle = U1.consume("B(b)");
  Matcher U2(Call2, Lex);
  auto Chunk1 = U2.consume("C(a, ");
  auto Chunk2 = U2.consume("{ b }");
  auto Chunk3 = U2.consume(")");

  EXPECT_THAT(std::move(Unexp).takeResult(),
              matchesLine(line({Chunk1, Middle, Chunk3})));
}

TEST_F(MacroCallReconstructorTest, StatementSequence) {
  auto Macros = createExpander({"SEMI=;"});
  Expansion Exp(Lex, *Macros);
  TokenList Call1 = Exp.expand("SEMI");
  TokenList Call2 = Exp.expand("SEMI");
  TokenList Call3 = Exp.expand("SEMI");

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line(E.consume(";")));
  EXPECT_TRUE(Unexp.finished());
  Unexp.addLine(line(E.consume(";")));
  EXPECT_TRUE(Unexp.finished());
  Unexp.addLine(line(E.consume(";")));
  EXPECT_TRUE(Unexp.finished());
  Matcher U1(Call1, Lex);
  Matcher U2(Call2, Lex);
  Matcher U3(Call3, Lex);
  EXPECT_THAT(std::move(Unexp).takeResult(),
              matchesLine(line(
                  {U1.consume("SEMI"),
                   children({line({U2.consume("SEMI"),
                                   children({line(U3.consume("SEMI"), 2)})},
                                  1)})})));
}

TEST_F(MacroCallReconstructorTest, NestedBlock) {
  auto Macros = createExpander({"ID(x)=x"});
  // Test: ID({ ID(a *b); })
  // 1. expand ID(a *b) -> a *b
  Expansion Exp1(Lex, *Macros);
  TokenList Call1 = Exp1.expand("ID", {"a *b"});
  // 2. expand ID({ a *b; })
  TokenList Arg;
  Arg.push_back(Lex.id("{"));
  Arg.append(Exp1.getTokens().begin(), Exp1.getTokens().end());
  Arg.push_back(Lex.id(";"));
  Arg.push_back(Lex.id("}"));
  Expansion Exp2(Lex, *Macros);
  TokenList Call2 = Exp2.expand("ID", {Arg});

  // Consume as-if formatted:
  // {
  //   a *b;
  // }
  UnexpandedMap Unexpanded =
      mergeUnexpanded(Exp1.getUnexpanded(), Exp2.getUnexpanded());
  MacroCallReconstructor Unexp(0, Unexpanded);
  Matcher E(Exp2.getTokens(), Lex);
  Unexp.addLine(line(E.consume("{")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("a *b;")));
  EXPECT_FALSE(Unexp.finished());
  Unexp.addLine(line(E.consume("}")));
  EXPECT_TRUE(Unexp.finished());

  // Expect lines:
  // ID({
  //   ID(a *b);
  // })
  Matcher U1(Call1, Lex);
  Matcher U2(Call2, Lex);
  auto Chunk2Start = U2.consume("ID(");
  auto Chunk2LBrace = U2.consume("{");
  U2.consume("a *b");
  auto Chunk2Mid = U2.consume(";");
  auto Chunk2RBrace = U2.consume("}");
  auto Chunk2End = U2.consume(")");
  auto Chunk1 = U1.consume("ID(a *b)");

  auto Expected = line({Chunk2Start,
                        children({
                            line(Chunk2LBrace, 1),
                            line({Chunk1, Chunk2Mid}, 1),
                            line(Chunk2RBrace, 1),
                        }),
                        Chunk2End});
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, NestedChildBlocks) {
  auto Macros = createExpander({"ID(x)=x", "CALL(x)=f([] { x })"});
  // Test: ID(CALL(CALL(return a * b;)))
  // 1. expand CALL(return a * b;)
  Expansion Exp1(Lex, *Macros);
  TokenList Call1 = Exp1.expand("CALL", {"return a * b;"});
  // 2. expand CALL(f([] { return a * b; }))
  Expansion Exp2(Lex, *Macros);
  TokenList Call2 = Exp2.expand("CALL", {Exp1.getTokens()});
  // 3. expand ID({ f([] { f([] { return a * b; }) }) })
  TokenList Arg3;
  Arg3.push_back(Lex.id("{"));
  Arg3.append(Exp2.getTokens().begin(), Exp2.getTokens().end());
  Arg3.push_back(Lex.id("}"));
  Expansion Exp3(Lex, *Macros);
  TokenList Call3 = Exp3.expand("ID", {Arg3});

  // Consume as-if formatted in three unwrapped lines:
  // 0: {
  // 1:   f([] {
  //        f([] {
  //          return a * b;
  //        })
  //      })
  // 2: }
  UnexpandedMap Unexpanded = mergeUnexpanded(
      Exp1.getUnexpanded(),
      mergeUnexpanded(Exp2.getUnexpanded(), Exp3.getUnexpanded()));
  MacroCallReconstructor Unexp(0, Unexpanded);
  Matcher E(Exp3.getTokens(), Lex);
  Unexp.addLine(line(E.consume("{")));
  Unexp.addLine(
      line({E.consume("f([] {"),
            children({line({E.consume("f([] {"),
                            children({line(E.consume("return a * b;"), 3)}),
                            E.consume("})")},
                           2)}),
            E.consume("})")},
           1));
  Unexp.addLine(line(E.consume("}")));
  EXPECT_TRUE(Unexp.finished());

  // Expect lines:
  // ID(
  //   {
  //   CALL(CALL(return a * b;))
  //   }
  // )
  Matcher U1(Call1, Lex);
  Matcher U2(Call2, Lex);
  Matcher U3(Call3, Lex);
  auto Chunk3Start = U3.consume("ID(");
  auto Chunk3LBrace = U3.consume("{");
  U3.consume("f([] { f([] { return a * b; }) })");
  auto Chunk3RBrace = U3.consume("}");
  auto Chunk3End = U3.consume(")");
  auto Chunk2Start = U2.consume("CALL(");
  U2.consume("f([] { return a * b; })");
  auto Chunk2End = U2.consume(")");
  auto Chunk1 = U1.consume("CALL(return a * b;)");

  auto Expected = line({
      Chunk3Start,
      children({
          line(Chunk3LBrace, 1),
          line(
              {
                  Chunk2Start,
                  Chunk1,
                  Chunk2End,
              },
              2),
          line(Chunk3RBrace, 1),
      }),
      Chunk3End,
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, NestedChildrenMultipleArguments) {
  auto Macros = createExpander({"CALL(a, b)=f([] { a; b; })"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("CALL", {std::string("int a"), "int b"});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line({
      E.consume("f([] {"),
      children({
          line(E.consume("int a;")),
          line(E.consume("int b;")),
      }),
      E.consume("})"),
  }));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line(U.consume("CALL(int a, int b)"));
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, ReverseOrderArgumentsInExpansion) {
  auto Macros = createExpander({"CALL(a, b)=b + a"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("CALL", {std::string("x"), "y"});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line(E.consume("y + x")));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line(U.consume("CALL(x, y)"));
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, MultipleToplevelUnwrappedLines) {
  auto Macros = createExpander({"ID(a, b)=a b"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("ID", {std::string("x; x"), "y"});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line(E.consume("x;")));
  Unexp.addLine(line(E.consume("x y")));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line({
      U.consume("ID("),
      children({
          line(U.consume("x;"), 1),
          line(U.consume("x"), 1),
      }),
      U.consume(", y)"),
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, NestedCallsMultipleLines) {
  auto Macros = createExpander({"ID(x)=x"});
  // Test: ID({ID(a * b);})
  // 1. expand ID(a * b)
  Expansion Exp1(Lex, *Macros);
  TokenList Call1 = Exp1.expand("ID", {"a * b"});
  // 2. expand ID({ a * b; })
  Expansion Exp2(Lex, *Macros);
  TokenList Arg2;
  Arg2.push_back(Lex.id("{"));
  Arg2.append(Exp1.getTokens().begin(), Exp1.getTokens().end());
  Arg2.push_back(Lex.id(";"));
  Arg2.push_back(Lex.id("}"));
  TokenList Call2 = Exp2.expand("ID", {Arg2});

  // Consume as-if formatted in three unwrapped lines:
  // 0: {
  // 1:   a * b;
  // 2: }
  UnexpandedMap Unexpanded =
      mergeUnexpanded(Exp1.getUnexpanded(), Exp2.getUnexpanded());
  MacroCallReconstructor Unexp(0, Unexpanded);
  Matcher E(Exp2.getTokens(), Lex);
  Unexp.addLine(line(E.consume("{")));
  Unexp.addLine(line(E.consume("a * b;")));
  Unexp.addLine(line(E.consume("}")));
  EXPECT_TRUE(Unexp.finished());

  // Expect lines:
  // ID(
  //     {
  //     ID(a * b);
  //     }
  // )
  Matcher U1(Call1, Lex);
  Matcher U2(Call2, Lex);
  auto Chunk2Start = U2.consume("ID(");
  auto Chunk2LBrace = U2.consume("{");
  U2.consume("a * b");
  auto Chunk2Semi = U2.consume(";");
  auto Chunk2RBrace = U2.consume("}");
  auto Chunk2End = U2.consume(")");
  auto Chunk1 = U1.consume("ID(a * b)");

  auto Expected = line({
      Chunk2Start,
      children({
          line({Chunk2LBrace}, 1),
          line({Chunk1, Chunk2Semi}, 1),
          line({Chunk2RBrace}, 1),
      }),
      Chunk2End,
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, ParentOutsideMacroCall) {
  auto Macros = createExpander({"ID(a)=a"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("ID", {std::string("x; y; z;")});

  auto Prefix = tokens("int a = []() {");
  auto Postfix = tokens("}();");
  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line({
      Prefix,
      children({
          line(E.consume("x;")),
          line(E.consume("y;")),
          line(E.consume("z;")),
      }),
      Postfix,
  }));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line({
      Prefix,
      children({
          line(
              {
                  U.consume("ID("),
                  children({
                      line(U.consume("x;"), 2),
                      line(U.consume("y;"), 2),
                      line(U.consume("z;"), 2),
                  }),
                  U.consume(")"),
              },
              1),
      }),
      Postfix,
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, ChildrenSplitAcrossArguments) {
  auto Macros = createExpander({"CALL(a, b)=f([]() a b)"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("CALL", {std::string("{ a;"), "b; }"});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line({
      E.consume("f([]() {"),
      children({
          line(E.consume("a;")),
          line(E.consume("b;")),
      }),
      E.consume("})"),
  }));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line({
      U.consume("CALL({"),
      children(line(U.consume("a;"), 1)),
      U.consume(", b; })"),
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, ChildrenAfterMacroCall) {
  auto Macros = createExpander({"CALL(a, b)=f([]() a b"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("CALL", {std::string("{ a"), "b"});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  auto Semi = tokens(";");
  auto SecondLine = tokens("c d;");
  auto ThirdLine = tokens("e f;");
  auto Postfix = tokens("})");
  Unexp.addLine(line({
      E.consume("f([]() {"),
      children({
          line({E.consume("a b"), Semi}),
          line(SecondLine),
          line(ThirdLine),
      }),
      Postfix,
  }));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line({
      U.consume("CALL({"),
      children(line(U.consume("a"), 1)),
      U.consume(", b)"),
      Semi,
      children(line(
          {
              SecondLine,
              children(line(
                  {
                      ThirdLine,
                      Postfix,
                  },
                  2)),
          },
          1)),
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, InvalidCodeSplittingBracesAcrossArgs) {
  auto Macros = createExpander({"M(a, b, c)=(a) (b) c"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("M", {std::string("{"), "x", ""});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  auto Prefix = tokens("({");
  Unexp.addLine(line({
      Prefix,
      children({
          line({
              E.consume("({"),
              children({line(E.consume(")(x)"))}),
          }),
      }),
  }));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line({
      Prefix,
      children({line(U.consume("M({,x,)"), 1)}),
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

TEST_F(MacroCallReconstructorTest, IndentLevelInExpandedCode) {
  auto Macros = createExpander({"ID(a)=a"});
  Expansion Exp(Lex, *Macros);
  TokenList Call = Exp.expand("ID", {std::string("[] { { x; } }")});

  MacroCallReconstructor Unexp(0, Exp.getUnexpanded());
  Matcher E(Exp.getTokens(), Lex);
  Unexp.addLine(line({
      E.consume("[] {"),
      children({
          line(E.consume("{"), 1),
          line(E.consume("x;"), 2),
          line(E.consume("}"), 1),
      }),
      E.consume("}"),
  }));
  EXPECT_TRUE(Unexp.finished());
  Matcher U(Call, Lex);
  auto Expected = line({
      U.consume("ID([] {"),
      children({
          line(U.consume("{"), 1),
          line(U.consume("x;"), 2),
          line(U.consume("}"), 1),
      }),
      U.consume("})"),
  });
  EXPECT_THAT(std::move(Unexp).takeResult(), matchesLine(Expected));
}

} // namespace
} // namespace format
} // namespace clang
