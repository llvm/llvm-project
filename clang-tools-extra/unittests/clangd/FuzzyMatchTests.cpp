//===-- FuzzyMatchTests.cpp - String fuzzy matcher tests --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FuzzyMatch.h"

#include "llvm/ADT/StringExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using testing::Not;

struct ExpectedMatch {
  // Annotations are optional, and will not be asserted if absent.
  ExpectedMatch(llvm::StringRef Match) : Word(Match), Annotated(Match) {
    for (char C : "[]")
      Word.erase(std::remove(Word.begin(), Word.end(), C), Word.end());
    if (Word.size() == Annotated->size())
      Annotated = llvm::None;
  }
  bool accepts(llvm::StringRef ActualAnnotated) const {
    return !Annotated || ActualAnnotated == *Annotated;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const ExpectedMatch &M) {
    OS << "'" << M.Word;
    if (M.Annotated)
      OS << "' as " << *M.Annotated;
    return OS;
  }

  std::string Word;

private:
  llvm::Optional<llvm::StringRef> Annotated;
};

struct MatchesMatcher : public testing::MatcherInterface<llvm::StringRef> {
  ExpectedMatch Candidate;
  llvm::Optional<float> Score;
  MatchesMatcher(ExpectedMatch Candidate, llvm::Optional<float> Score)
      : Candidate(std::move(Candidate)), Score(Score) {}

  void DescribeTo(::std::ostream *OS) const override {
    llvm::raw_os_ostream(*OS) << "Matches " << Candidate;
    if (Score)
      *OS << " with score " << *Score;
  }

  bool MatchAndExplain(llvm::StringRef Pattern,
                       testing::MatchResultListener *L) const override {
    std::unique_ptr<llvm::raw_ostream> OS(
        L->stream()
            ? (llvm::raw_ostream *)(new llvm::raw_os_ostream(*L->stream()))
            : new llvm::raw_null_ostream());
    FuzzyMatcher Matcher(Pattern);
    auto Result = Matcher.match(Candidate.Word);
    auto AnnotatedMatch = Matcher.dumpLast(*OS << "\n");
    return Result && Candidate.accepts(AnnotatedMatch) &&
           (!Score || testing::Value(*Result, testing::FloatEq(*Score)));
  }
};

// Accepts patterns that match a given word, optionally requiring a score.
// Dumps the debug tables on match failure.
testing::Matcher<llvm::StringRef> matches(llvm::StringRef M,
                                          llvm::Optional<float> Score = {}) {
  return testing::MakeMatcher<llvm::StringRef>(new MatchesMatcher(M, Score));
}

TEST(FuzzyMatch, Matches) {
  EXPECT_THAT("", matches("unique_ptr"));
  EXPECT_THAT("u_p", matches("[u]nique[_p]tr"));
  EXPECT_THAT("up", matches("[u]nique_[p]tr"));
  EXPECT_THAT("uq", Not(matches("unique_ptr")));
  EXPECT_THAT("qp", Not(matches("unique_ptr")));
  EXPECT_THAT("log", Not(matches("SVGFEMorphologyElement")));

  EXPECT_THAT("tit", matches("win.[tit]"));
  EXPECT_THAT("title", matches("win.[title]"));
  EXPECT_THAT("WordCla", matches("[Word]Character[Cla]ssifier"));
  EXPECT_THAT("WordCCla", matches("[WordC]haracter[Cla]ssifier"));

  EXPECT_THAT("dete", Not(matches("editor.quickSuggestionsDelay")));

  EXPECT_THAT("highlight", matches("editorHover[Highlight]"));
  EXPECT_THAT("hhighlight", matches("editor[H]over[Highlight]"));
  EXPECT_THAT("dhhighlight", Not(matches("editorHoverHighlight")));

  EXPECT_THAT("-moz", matches("[-moz]-foo"));
  EXPECT_THAT("moz", matches("-[moz]-foo"));
  EXPECT_THAT("moza", matches("-[moz]-[a]nimation"));

  EXPECT_THAT("ab", matches("[ab]A"));
  EXPECT_THAT("ccm", Not(matches("cacmelCase")));
  EXPECT_THAT("bti", Not(matches("the_black_knight")));
  EXPECT_THAT("ccm", Not(matches("camelCase")));
  EXPECT_THAT("cmcm", Not(matches("camelCase")));
  EXPECT_THAT("BK", matches("the_[b]lack_[k]night"));
  EXPECT_THAT("KeyboardLayout=", Not(matches("KeyboardLayout")));
  EXPECT_THAT("LLL", matches("SVisual[L]ogger[L]ogs[L]ist"));
  EXPECT_THAT("LLLL", Not(matches("SVilLoLosLi")));
  EXPECT_THAT("LLLL", Not(matches("SVisualLoggerLogsList")));
  EXPECT_THAT("TEdit", matches("[T]ext[Edit]"));
  EXPECT_THAT("TEdit", matches("[T]ext[Edit]or"));
  EXPECT_THAT("TEdit", Not(matches("[T]ext[edit]")));
  EXPECT_THAT("TEdit", matches("[t]ext_[edit]"));
  EXPECT_THAT("TEditDt", matches("[T]ext[Edit]or[D]ecoration[T]ype"));
  EXPECT_THAT("TEdit", matches("[T]ext[Edit]orDecorationType"));
  EXPECT_THAT("Tedit", matches("[T]ext[Edit]"));
  EXPECT_THAT("ba", Not(matches("?AB?")));
  EXPECT_THAT("bkn", matches("the_[b]lack_[kn]ight"));
  EXPECT_THAT("bt", Not(matches("the_[b]lack_knigh[t]")));
  EXPECT_THAT("ccm", Not(matches("[c]amelCase[cm]")));
  EXPECT_THAT("fdm", Not(matches("[f]in[dM]odel")));
  EXPECT_THAT("fob", Not(matches("[fo]o[b]ar")));
  EXPECT_THAT("fobz", Not(matches("foobar")));
  EXPECT_THAT("foobar", matches("[foobar]"));
  EXPECT_THAT("form", matches("editor.[form]atOnSave"));
  EXPECT_THAT("g p", matches("[G]it:[ P]ull"));
  EXPECT_THAT("g p", matches("[G]it:[ P]ull"));
  EXPECT_THAT("gip", matches("[Gi]t: [P]ull"));
  EXPECT_THAT("gip", matches("[Gi]t: [P]ull"));
  EXPECT_THAT("gp", matches("[G]it: [P]ull"));
  EXPECT_THAT("gp", matches("[G]it_Git_[P]ull"));
  EXPECT_THAT("is", matches("[I]mport[S]tatement"));
  EXPECT_THAT("is", matches("[is]Valid"));
  EXPECT_THAT("lowrd", Not(matches("[low]Wo[rd]")));
  EXPECT_THAT("myvable", Not(matches("[myva]ria[ble]")));
  EXPECT_THAT("no", Not(matches("")));
  EXPECT_THAT("no", Not(matches("match")));
  EXPECT_THAT("ob", Not(matches("foobar")));
  EXPECT_THAT("sl", matches("[S]Visual[L]oggerLogsList"));
  EXPECT_THAT("sllll", matches("[S]Visua[L]ogger[Ll]ama[L]ist"));
  EXPECT_THAT("THRE", matches("H[T]ML[HRE]lement"));
  EXPECT_THAT("b", Not(matches("NDEBUG")));
  EXPECT_THAT("Three", matches("[Three]"));
  EXPECT_THAT("fo", Not(matches("barfoo")));
  EXPECT_THAT("fo", matches("bar_[fo]o"));
  EXPECT_THAT("fo", matches("bar_[Fo]o"));
  EXPECT_THAT("fo", matches("bar [fo]o"));
  EXPECT_THAT("fo", matches("bar.[fo]o"));
  EXPECT_THAT("fo", matches("bar/[fo]o"));
  EXPECT_THAT("fo", matches("bar\\[fo]o"));

  EXPECT_THAT(
      "aaaaaa",
      matches("[aaaaaa]aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
  EXPECT_THAT("baba", Not(matches("ababababab")));
  EXPECT_THAT("fsfsfs", Not(matches("dsafdsafdsafdsafdsafdsafdsafasdfdsa")));
  EXPECT_THAT("fsfsfsfsfsfsfsf",
              Not(matches("dsafdsafdsafdsafdsafdsafdsafasdfdsafdsafdsafdsafdsfd"
                          "safdsfdfdfasdnfdsajfndsjnafjndsajlknfdsa")));

  EXPECT_THAT("  g", matches("[  g]roup"));
  EXPECT_THAT("g", matches("  [g]roup"));
  EXPECT_THAT("g g", Not(matches("  groupGroup")));
  EXPECT_THAT("g g", matches("  [g]roup[ G]roup"));
  EXPECT_THAT(" g g", matches("[ ] [g]roup[ G]roup"));
  EXPECT_THAT("zz", matches("[zz]Group"));
  EXPECT_THAT("zzg", matches("[zzG]roup"));
  EXPECT_THAT("g", matches("zz[G]roup"));

  EXPECT_THAT("aaaa", matches("_a_[aaaa]")); // Prefer consecutive.
  // These would ideally match, but would need special segmentation rules.
  EXPECT_THAT("printf", Not(matches("s[printf]")));
  EXPECT_THAT("str", Not(matches("o[str]eam")));
  EXPECT_THAT("strcpy", Not(matches("strncpy")));
  EXPECT_THAT("std", Not(matches("PTHREAD_MUTEX_STALLED")));
  EXPECT_THAT("std", Not(matches("pthread_condattr_setpshared")));
}

struct RankMatcher : public testing::MatcherInterface<llvm::StringRef> {
  std::vector<ExpectedMatch> RankedStrings;
  RankMatcher(std::initializer_list<ExpectedMatch> RankedStrings)
      : RankedStrings(RankedStrings) {}

  void DescribeTo(::std::ostream *OS) const override {
    llvm::raw_os_ostream O(*OS);
    O << "Ranks strings in order: [";
    for (const auto &Str : RankedStrings)
      O << "\n\t" << Str;
    O << "\n]";
  }

  bool MatchAndExplain(llvm::StringRef Pattern,
                       testing::MatchResultListener *L) const override {
    std::unique_ptr<llvm::raw_ostream> OS(
        L->stream()
            ? (llvm::raw_ostream *)(new llvm::raw_os_ostream(*L->stream()))
            : new llvm::raw_null_ostream());
    FuzzyMatcher Matcher(Pattern);
    const ExpectedMatch *LastMatch;
    llvm::Optional<float> LastScore;
    bool Ok = true;
    for (const auto &Str : RankedStrings) {
      auto Score = Matcher.match(Str.Word);
      if (!Score) {
        *OS << "\nDoesn't match '" << Str.Word << "'";
        Matcher.dumpLast(*OS << "\n");
        Ok = false;
      } else {
        std::string Buf;
        llvm::raw_string_ostream Info(Buf);
        auto AnnotatedMatch = Matcher.dumpLast(Info);

        if (!Str.accepts(AnnotatedMatch)) {
          *OS << "\nDoesn't match " << Str << ", but " << AnnotatedMatch << "\n"
              << Info.str();
          Ok = false;
        } else if (LastScore && *LastScore < *Score) {
          *OS << "\nRanks '" << Str.Word << "'=" << *Score << " above '"
              << LastMatch->Word << "'=" << *LastScore << "\n"
              << Info.str();
          Matcher.match(LastMatch->Word);
          Matcher.dumpLast(*OS << "\n");
          Ok = false;
        }
      }
      LastMatch = &Str;
      LastScore = Score;
    }
    return Ok;
  }
};

// Accepts patterns that match all the strings and rank them in the given order.
// Dumps the debug tables on match failure.
template <typename... T>
testing::Matcher<llvm::StringRef> ranks(T... RankedStrings) {
  return testing::MakeMatcher<llvm::StringRef>(
      new RankMatcher{ExpectedMatch(RankedStrings)...});
}

TEST(FuzzyMatch, Ranking) {
  EXPECT_THAT("cons",
              ranks("[cons]ole", "[Cons]ole", "ArrayBuffer[Cons]tructor"));
  EXPECT_THAT("foo", ranks("[foo]", "[Foo]"));
  EXPECT_THAT("onMes",
              ranks("[onMes]sage", "[onmes]sage", "[on]This[M]ega[Es]capes"));
  EXPECT_THAT("CC", ranks("[C]amel[C]ase", "[c]amel[C]ase"));
  EXPECT_THAT("cC", ranks("[c]amel[C]ase", "[C]amel[C]ase"));
  EXPECT_THAT("p", ranks("[p]", "[p]arse", "[p]osix", "[p]afdsa", "[p]ath"));
  EXPECT_THAT("pa", ranks("[pa]rse", "[pa]th", "[pa]fdsa"));
  EXPECT_THAT("log", ranks("[log]", "Scroll[Log]icalPosition"));
  EXPECT_THAT("e", ranks("[e]lse", "Abstract[E]lement"));
  EXPECT_THAT("workbench.sideb",
              ranks("[workbench.sideB]ar.location",
                    "[workbench.]editor.default[SideB]ySideLayout"));
  EXPECT_THAT("editor.r", ranks("[editor.r]enderControlCharacter",
                                "[editor.]overview[R]ulerlanes",
                                "diff[Editor.r]enderSideBySide"));
  EXPECT_THAT("-mo", ranks("[-mo]z-columns", "[-]ms-ime-[mo]de"));
  EXPECT_THAT("convertModelPosition",
              ranks("[convertModelPosition]ToViewPosition",
                    "[convert]ViewTo[ModelPosition]"));
  EXPECT_THAT("is", ranks("[is]ValidViewletId", "[i]mport [s]tatement"));
  EXPECT_THAT("strcpy", ranks("[strcpy]", "[strcpy]_s"));
}

// Verify some bounds so we know scores fall in the right range.
// Testing exact scores is fragile, so we prefer Ranking tests.
TEST(FuzzyMatch, Scoring) {
  EXPECT_THAT("abs", matches("[a]w[B]xYz[S]", 0.f));
  EXPECT_THAT("abs", matches("[abs]l", 1.f));
  EXPECT_THAT("abs", matches("[abs]", 2.f));
  EXPECT_THAT("Abs", matches("[abs]", 2.f));
}

// Returns pretty-printed segmentation of Text.
// e.g. std::basic_string --> +--  +---- +-----
std::string segment(llvm::StringRef Text) {
  std::vector<CharRole> Roles(Text.size());
  calculateRoles(Text, Roles);
  std::string Printed;
  for (unsigned I = 0; I < Text.size(); ++I)
    Printed.push_back("?-+ "[static_cast<unsigned>(Roles[I])]);
  return Printed;
}

// this is a no-op hack so clang-format will vertically align our testcases.
llvm::StringRef returns(llvm::StringRef Text) { return Text; }

TEST(FuzzyMatch, Segmentation) {
  EXPECT_THAT(segment("std::basic_string"), //
              returns("+--  +---- +-----"));
  EXPECT_THAT(segment("XMLHttpRequest"), //
              returns("+--+---+------"));
  EXPECT_THAT(segment("t3h PeNgU1N oF d00m!!!!!!!!"), //
              returns("+-- +-+-+-+ ++ +---        "));
}

} // namespace
} // namespace clangd
} // namespace clang
