//===-- FuzzyMatchTests.cpp - String fuzzy matcher tests --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FuzzyMatch.h"

#include "llvm/ADT/StringExtras.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using namespace llvm;
using testing::Not;

struct ExpectedMatch {
  ExpectedMatch(StringRef Annotated) : Word(Annotated), Annotated(Annotated) {
    for (char C : "[]")
      Word.erase(std::remove(Word.begin(), Word.end(), C), Word.end());
  }
  std::string Word;
  StringRef Annotated;
};
raw_ostream &operator<<(raw_ostream &OS, const ExpectedMatch &M) {
  return OS << "'" << M.Word << "' as " << M.Annotated;
}

struct MatchesMatcher : public testing::MatcherInterface<StringRef> {
  ExpectedMatch Candidate;
  MatchesMatcher(ExpectedMatch Candidate) : Candidate(std::move(Candidate)) {}

  void DescribeTo(::std::ostream *OS) const override {
    raw_os_ostream(*OS) << "Matches " << Candidate;
  }

  bool MatchAndExplain(StringRef Pattern,
                       testing::MatchResultListener *L) const override {
    std::unique_ptr<raw_ostream> OS(
        L->stream() ? (raw_ostream *)(new raw_os_ostream(*L->stream()))
                    : new raw_null_ostream());
    FuzzyMatcher Matcher(Pattern);
    auto Result = Matcher.match(Candidate.Word);
    auto AnnotatedMatch = Matcher.dumpLast(*OS << "\n");
    return Result && AnnotatedMatch == Candidate.Annotated;
  }
};

// Accepts patterns that match a given word.
// Dumps the debug tables on match failure.
testing::Matcher<StringRef> matches(StringRef M) {
  return testing::MakeMatcher<StringRef>(new MatchesMatcher(M));
}

TEST(FuzzyMatch, Matches) {
  EXPECT_THAT("u_p", matches("[u]nique[_p]tr"));
  EXPECT_THAT("up", matches("[u]nique_[p]tr"));
  EXPECT_THAT("uq", matches("[u]ni[q]ue_ptr"));
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
  EXPECT_THAT("ccm", matches("[c]a[cm]elCase"));
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
  EXPECT_THAT("TEdit", matches("[Te]xte[dit]"));
  EXPECT_THAT("TEdit", matches("[t]ext_[edit]"));
  EXPECT_THAT("TEditDit", matches("[T]ext[Edit]or[D]ecorat[i]on[T]ype"));
  EXPECT_THAT("TEdit", matches("[T]ext[Edit]orDecorationType"));
  EXPECT_THAT("Tedit", matches("[T]ext[Edit]"));
  EXPECT_THAT("ba", Not(matches("?AB?")));
  EXPECT_THAT("bkn", matches("the_[b]lack_[kn]ight"));
  EXPECT_THAT("bt", matches("the_[b]lack_knigh[t]"));
  EXPECT_THAT("ccm", matches("[c]amelCase[cm]"));
  EXPECT_THAT("fdm", matches("[f]in[dM]odel"));
  EXPECT_THAT("fob", matches("[fo]o[b]ar"));
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
  EXPECT_THAT("lowrd", matches("[low]Wo[rd]"));
  EXPECT_THAT("myvable", matches("[myva]ria[ble]"));
  EXPECT_THAT("no", Not(matches("")));
  EXPECT_THAT("no", Not(matches("match")));
  EXPECT_THAT("ob", Not(matches("foobar")));
  EXPECT_THAT("sl", matches("[S]Visual[L]oggerLogsList"));
  EXPECT_THAT("sllll", matches("[S]Visua[lL]ogger[L]ogs[L]ist"));
  EXPECT_THAT("Three", matches("H[T]ML[HRE]l[e]ment"));
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
  EXPECT_THAT("printf", matches("s[printf]"));
  EXPECT_THAT("str", matches("o[str]eam"));
}

struct RankMatcher : public testing::MatcherInterface<StringRef> {
  std::vector<ExpectedMatch> RankedStrings;
  RankMatcher(std::initializer_list<ExpectedMatch> RankedStrings)
      : RankedStrings(RankedStrings) {}

  void DescribeTo(::std::ostream *OS) const override {
    raw_os_ostream O(*OS);
    O << "Ranks strings in order: [";
    for (const auto &Str : RankedStrings)
      O << "\n\t" << Str;
    O << "\n]";
  }

  bool MatchAndExplain(StringRef Pattern,
                       testing::MatchResultListener *L) const override {
    std::unique_ptr<raw_ostream> OS(
        L->stream() ? (raw_ostream *)(new raw_os_ostream(*L->stream()))
                    : new raw_null_ostream());
    FuzzyMatcher Matcher(Pattern);
    const ExpectedMatch *LastMatch;
    Optional<float> LastScore;
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

        if (AnnotatedMatch != Str.Annotated) {
          *OS << "\nMatched " << Str.Word << " as " << AnnotatedMatch
              << " instead of " << Str.Annotated << "\n"
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
template <typename... T> testing::Matcher<StringRef> ranks(T... RankedStrings) {
  return testing::MakeMatcher<StringRef>(
      new RankMatcher{ExpectedMatch(RankedStrings)...});
}

TEST(FuzzyMatch, Ranking) {
  EXPECT_THAT("eb", ranks("[e]mplace_[b]ack", "[e]m[b]ed"));
  EXPECT_THAT("cons",
              ranks("[cons]ole", "[Cons]ole", "ArrayBuffer[Cons]tructor"));
  EXPECT_THAT("foo", ranks("[foo]", "[Foo]"));
  EXPECT_THAT("onMess",
              ranks("[onMess]age", "[onmess]age", "[on]This[M]ega[Es]cape[s]"));
  EXPECT_THAT("CC", ranks("[C]amel[C]ase", "[c]amel[C]ase"));
  EXPECT_THAT("cC", ranks("[c]amel[C]ase", "[C]amel[C]ase"));
  EXPECT_THAT("p", ranks("[p]arse", "[p]osix", "[p]afdsa", "[p]ath", "[p]"));
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
  EXPECT_THAT("title", ranks("window.[title]",
                             "files.[t]r[i]m[T]rai[l]ingWhit[e]space"));
  EXPECT_THAT("strcpy", ranks("[strcpy]", "[strcpy]_s", "[str]n[cpy]"));
  EXPECT_THAT("close", ranks("workbench.quickOpen.[close]OnFocusOut",
                             "[c]ss.[l]int.imp[o]rt[S]tat[e]ment",
                             "[c]ss.co[lo]rDecorator[s].[e]nable"));
}

} // namespace
} // namespace clangd
} // namespace clang
