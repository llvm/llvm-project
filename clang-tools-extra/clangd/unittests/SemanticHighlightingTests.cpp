//==- SemanticHighlightingTests.cpp - SemanticHighlighting tests-*- C++ -* -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

MATCHER_P(LineNumber, L, "") { return arg.Line == L; }
MATCHER(EmptyHighlightings, "") { return arg.Tokens.empty(); }

std::vector<HighlightingToken>
makeHighlightingTokens(llvm::ArrayRef<Range> Ranges, HighlightingKind Kind) {
  std::vector<HighlightingToken> Tokens(Ranges.size());
  for (int I = 0, End = Ranges.size(); I < End; ++I) {
    Tokens[I].R = Ranges[I];
    Tokens[I].Kind = Kind;
  }

  return Tokens;
}

std::vector<HighlightingToken> getExpectedTokens(Annotations &Test) {
  static const std::map<HighlightingKind, std::string> KindToString{
      {HighlightingKind::Variable, "Variable"},
      {HighlightingKind::LocalVariable, "LocalVariable"},
      {HighlightingKind::Parameter, "Parameter"},
      {HighlightingKind::Function, "Function"},
      {HighlightingKind::Class, "Class"},
      {HighlightingKind::Enum, "Enum"},
      {HighlightingKind::Namespace, "Namespace"},
      {HighlightingKind::EnumConstant, "EnumConstant"},
      {HighlightingKind::Field, "Field"},
      {HighlightingKind::StaticField, "StaticField"},
      {HighlightingKind::Method, "Method"},
      {HighlightingKind::StaticMethod, "StaticMethod"},
      {HighlightingKind::Typedef, "Typedef"},
      {HighlightingKind::DependentType, "DependentType"},
      {HighlightingKind::DependentName, "DependentName"},
      {HighlightingKind::TemplateParameter, "TemplateParameter"},
      {HighlightingKind::Primitive, "Primitive"},
      {HighlightingKind::Macro, "Macro"}};
  std::vector<HighlightingToken> ExpectedTokens;
  for (const auto &KindString : KindToString) {
    std::vector<HighlightingToken> Toks = makeHighlightingTokens(
        Test.ranges(KindString.second), KindString.first);
    ExpectedTokens.insert(ExpectedTokens.end(), Toks.begin(), Toks.end());
  }
  llvm::sort(ExpectedTokens);
  return ExpectedTokens;
}

/// Annotates the input code with provided semantic highlightings. Results look
/// something like:
///   class $Class[[X]] {
///     $Primitive[[int]] $Field[[a]] = 0;
///   };
std::string annotate(llvm::StringRef Input,
                     llvm::ArrayRef<HighlightingToken> Tokens) {
  assert(std::is_sorted(
      Tokens.begin(), Tokens.end(),
      [](const HighlightingToken &L, const HighlightingToken &R) {
        return L.R.start < R.R.start;
      }));

  std::string Result;
  unsigned NextChar = 0;
  for (auto &T : Tokens) {
    unsigned StartOffset = llvm::cantFail(positionToOffset(Input, T.R.start));
    unsigned EndOffset = llvm::cantFail(positionToOffset(Input, T.R.end));
    assert(StartOffset <= EndOffset);
    assert(NextChar <= StartOffset);

    Result += Input.substr(NextChar, StartOffset - NextChar);
    Result += llvm::formatv("${0}[[{1}]]", T.Kind,
                            Input.substr(StartOffset, EndOffset - StartOffset));
    NextChar = EndOffset;
  }
  Result += Input.substr(NextChar);
  return Result;
}

void checkHighlightings(llvm::StringRef Code,
                        std::vector<std::pair</*FileName*/ llvm::StringRef,
                                              /*FileContent*/ llvm::StringRef>>
                            AdditionalFiles = {}) {
  Annotations Test(Code);
  TestTU TU;
  TU.Code = Test.code();

  // FIXME: Auto-completion in a template requires disabling delayed template
  // parsing.
  TU.ExtraArgs.push_back("-fno-delayed-template-parsing");

  for (auto File : AdditionalFiles)
    TU.AdditionalFiles.insert({File.first, File.second});
  auto AST = TU.build();

  EXPECT_EQ(Code, annotate(Test.code(), getSemanticHighlightings(AST)));
}

// Any annotations in OldCode and NewCode are converted into their corresponding
// HighlightingToken. The tokens are diffed against each other. Any lines where
// the tokens should diff must be marked with a ^ somewhere on that line in
// NewCode. If there are diffs that aren't marked with ^ the test fails. The
// test also fails if there are lines marked with ^ that don't differ.
void checkDiffedHighlights(llvm::StringRef OldCode, llvm::StringRef NewCode) {
  Annotations OldTest(OldCode);
  Annotations NewTest(NewCode);
  std::vector<HighlightingToken> OldTokens = getExpectedTokens(OldTest);
  std::vector<HighlightingToken> NewTokens = getExpectedTokens(NewTest);

  llvm::DenseMap<int, std::vector<HighlightingToken>> ExpectedLines;
  for (const Position &Point : NewTest.points()) {
    ExpectedLines[Point.line]; // Default initialize to an empty line. Tokens
                               // are inserted on these lines later.
  }
  std::vector<LineHighlightings> ExpectedLinePairHighlighting;
  for (const HighlightingToken &Token : NewTokens) {
    auto It = ExpectedLines.find(Token.R.start.line);
    if (It != ExpectedLines.end())
      It->second.push_back(Token);
  }
  for (auto &LineTokens : ExpectedLines)
    ExpectedLinePairHighlighting.push_back(
        {LineTokens.first, LineTokens.second, /*IsInactive = */ false});

  std::vector<LineHighlightings> ActualDiffed =
      diffHighlightings(NewTokens, OldTokens);
  EXPECT_THAT(ActualDiffed,
              testing::UnorderedElementsAreArray(ExpectedLinePairHighlighting))
      << OldCode;
}

TEST(SemanticHighlighting, GetsCorrectTokens) {
  const char *TestCases[] = {
      R"cpp(
      struct $Class[[AS]] {
        double $Field[[SomeMember]];
      };
      struct {
      } $Variable[[S]];
      void $Function[[foo]](int $Parameter[[A]], $Class[[AS]] $Parameter[[As]]) {
        $Primitive[[auto]] $LocalVariable[[VeryLongVariableName]] = 12312;
        $Class[[AS]]     $LocalVariable[[AA]];
        $Primitive[[auto]] $LocalVariable[[L]] = $LocalVariable[[AA]].$Field[[SomeMember]] + $Parameter[[A]];
        auto $LocalVariable[[FN]] = [ $LocalVariable[[AA]]](int $Parameter[[A]]) -> void {};
        $LocalVariable[[FN]](12312);
      }
    )cpp",
      R"cpp(
      void $Function[[foo]](int);
      void $Function[[Gah]]();
      void $Function[[foo]]() {
        auto $LocalVariable[[Bou]] = $Function[[Gah]];
      }
      struct $Class[[A]] {
        void $Method[[abc]]();
      };
    )cpp",
      R"cpp(
      namespace $Namespace[[abc]] {
        template<typename $TemplateParameter[[T]]>
        struct $Class[[A]] {
          $TemplateParameter[[T]] $Field[[t]];
        };
      }
      template<typename $TemplateParameter[[T]]>
      struct $Class[[C]] : $Namespace[[abc]]::$Class[[A]]<$TemplateParameter[[T]]> {
        typename $TemplateParameter[[T]]::$DependentType[[A]]* $Field[[D]];
      };
      $Namespace[[abc]]::$Class[[A]]<int> $Variable[[AA]];
      typedef $Namespace[[abc]]::$Class[[A]]<int> $Class[[AAA]];
      struct $Class[[B]] {
        $Class[[B]]();
        ~$Class[[B]]();
        void operator<<($Class[[B]]);
        $Class[[AAA]] $Field[[AA]];
      };
      $Class[[B]]::$Class[[B]]() {}
      $Class[[B]]::~$Class[[B]]() {}
      void $Function[[f]] () {
        $Class[[B]] $LocalVariable[[BB]] = $Class[[B]]();
        $LocalVariable[[BB]].~$Class[[B]]();
        $Class[[B]]();
      }
    )cpp",
      R"cpp(
      enum class $Enum[[E]] {
        $EnumConstant[[A]],
        $EnumConstant[[B]],
      };
      enum $Enum[[EE]] {
        $EnumConstant[[Hi]],
      };
      struct $Class[[A]] {
        $Enum[[E]] $Field[[EEE]];
        $Enum[[EE]] $Field[[EEEE]];
      };
      int $Variable[[I]] = $EnumConstant[[Hi]];
      $Enum[[E]] $Variable[[L]] = $Enum[[E]]::$EnumConstant[[B]];
    )cpp",
      R"cpp(
      namespace $Namespace[[abc]] {
        namespace {}
        namespace $Namespace[[bcd]] {
          struct $Class[[A]] {};
          namespace $Namespace[[cde]] {
            struct $Class[[A]] {
              enum class $Enum[[B]] {
                $EnumConstant[[Hi]],
              };
            };
          }
        }
      }
      using namespace $Namespace[[abc]]::$Namespace[[bcd]];
      namespace $Namespace[[vwz]] =
            $Namespace[[abc]]::$Namespace[[bcd]]::$Namespace[[cde]];
      $Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable[[AA]];
      $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]] $Variable[[AAA]] =
            $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]]::$EnumConstant[[Hi]];
      ::$Namespace[[vwz]]::$Class[[A]] $Variable[[B]];
      ::$Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable[[BB]];
    )cpp",
      R"cpp(
      struct $Class[[D]] {
        double $Field[[C]];
      };
      struct $Class[[A]] {
        double $Field[[B]];
        $Class[[D]] $Field[[E]];
        static double $StaticField[[S]];
        static void $StaticMethod[[bar]]() {}
        void $Method[[foo]]() {
          $Field[[B]] = 123;
          this->$Field[[B]] = 156;
          this->$Method[[foo]]();
          $Method[[foo]]();
          $StaticMethod[[bar]]();
          $StaticField[[S]] = 90.1;
        }
      };
      void $Function[[foo]]() {
        $Class[[A]] $LocalVariable[[AA]];
        $LocalVariable[[AA]].$Field[[B]] += 2;
        $LocalVariable[[AA]].$Method[[foo]]();
        $LocalVariable[[AA]].$Field[[E]].$Field[[C]];
        $Class[[A]]::$StaticField[[S]] = 90;
      }
    )cpp",
      R"cpp(
      struct $Class[[AA]] {
        int $Field[[A]];
      }
      int $Variable[[B]];
      $Class[[AA]] $Variable[[A]]{$Variable[[B]]};
    )cpp",
      R"cpp(
      namespace $Namespace[[a]] {
        struct $Class[[A]] {};
        typedef char $Primitive[[C]];
      }
      typedef $Namespace[[a]]::$Class[[A]] $Class[[B]];
      using $Class[[BB]] = $Namespace[[a]]::$Class[[A]];
      enum class $Enum[[E]] {};
      typedef $Enum[[E]] $Enum[[C]];
      typedef $Enum[[C]] $Enum[[CC]];
      using $Enum[[CD]] = $Enum[[CC]];
      $Enum[[CC]] $Function[[f]]($Class[[B]]);
      $Enum[[CD]] $Function[[f]]($Class[[BB]]);
      typedef $Namespace[[a]]::$Primitive[[C]] $Primitive[[PC]];
      typedef float $Primitive[[F]];
    )cpp",
      R"cpp(
      template<typename $TemplateParameter[[T]], typename = void>
      class $Class[[A]] {
        $TemplateParameter[[T]] $Field[[AA]];
        $TemplateParameter[[T]] $Method[[foo]]();
      };
      template<class $TemplateParameter[[TT]]>
      class $Class[[B]] {
        $Class[[A]]<$TemplateParameter[[TT]]> $Field[[AA]];
      };
      template<class $TemplateParameter[[TT]], class $TemplateParameter[[GG]]>
      class $Class[[BB]] {};
      template<class $TemplateParameter[[T]]>
      class $Class[[BB]]<$TemplateParameter[[T]], int> {};
      template<class $TemplateParameter[[T]]>
      class $Class[[BB]]<$TemplateParameter[[T]], $TemplateParameter[[T]]*> {};

      template<template<class> class $TemplateParameter[[T]], class $TemplateParameter[[C]]>
      $TemplateParameter[[T]]<$TemplateParameter[[C]]> $Function[[f]]();

      template<typename>
      class $Class[[Foo]] {};

      template<typename $TemplateParameter[[T]]>
      void $Function[[foo]]($TemplateParameter[[T]] ...);
    )cpp",
      R"cpp(
      template <class $TemplateParameter[[T]]>
      struct $Class[[Tmpl]] {$TemplateParameter[[T]] $Field[[x]] = 0;};
      extern template struct $Class[[Tmpl]]<float>;
      template struct $Class[[Tmpl]]<double>;
    )cpp",
      // This test is to guard against highlightings disappearing when using
      // conversion operators as their behaviour in the clang AST differ from
      // other CXXMethodDecls.
      R"cpp(
      class $Class[[Foo]] {};
      struct $Class[[Bar]] {
        explicit operator $Class[[Foo]]*() const;
        explicit operator int() const;
        operator $Class[[Foo]]();
      };
      void $Function[[f]]() {
        $Class[[Bar]] $LocalVariable[[B]];
        $Class[[Foo]] $LocalVariable[[F]] = $LocalVariable[[B]];
        $Class[[Foo]] *$LocalVariable[[FP]] = ($Class[[Foo]]*)$LocalVariable[[B]];
        int $LocalVariable[[I]] = (int)$LocalVariable[[B]];
      }
    )cpp"
      R"cpp(
      struct $Class[[B]] {};
      struct $Class[[A]] {
        $Class[[B]] $Field[[BB]];
        $Class[[A]] &operator=($Class[[A]] &&$Parameter[[O]]);
      };

      $Class[[A]] &$Class[[A]]::operator=($Class[[A]] &&$Parameter[[O]]) = default;
    )cpp",
      R"cpp(
      enum $Enum[[En]] {
        $EnumConstant[[EC]],
      };
      class $Class[[Foo]] {};
      class $Class[[Bar]] {
        $Class[[Foo]] $Field[[Fo]];
        $Enum[[En]] $Field[[E]];
        int $Field[[I]];
        $Class[[Bar]] ($Class[[Foo]] $Parameter[[F]],
                $Enum[[En]] $Parameter[[E]])
        : $Field[[Fo]] ($Parameter[[F]]), $Field[[E]] ($Parameter[[E]]),
          $Field[[I]] (123) {}
      };
      class $Class[[Bar2]] : public $Class[[Bar]] {
        $Class[[Bar2]]() : $Class[[Bar]]($Class[[Foo]](), $EnumConstant[[EC]]) {}
      };
    )cpp",
      R"cpp(
      enum $Enum[[E]] {
        $EnumConstant[[E]],
      };
      class $Class[[Foo]] {};
      $Enum[[auto]] $Variable[[AE]] = $Enum[[E]]::$EnumConstant[[E]];
      $Class[[auto]] $Variable[[AF]] = $Class[[Foo]]();
      $Class[[decltype]](auto) $Variable[[AF2]] = $Class[[Foo]]();
      $Class[[auto]] *$Variable[[AFP]] = &$Variable[[AF]];
      $Enum[[auto]] &$Variable[[AER]] = $Variable[[AE]];
      $Primitive[[auto]] $Variable[[Form]] = 10.2 + 2 * 4;
      $Primitive[[decltype]]($Variable[[Form]]) $Variable[[F]] = 10;
      auto $Variable[[Fun]] = []()->void{};
    )cpp",
      R"cpp(
      class $Class[[G]] {};
      template<$Class[[G]] *$TemplateParameter[[U]]>
      class $Class[[GP]] {};
      template<$Class[[G]] &$TemplateParameter[[U]]>
      class $Class[[GR]] {};
      template<int *$TemplateParameter[[U]]>
      class $Class[[IP]] {
        void $Method[[f]]() {
          *$TemplateParameter[[U]] += 5;
        }
      };
      template<unsigned $TemplateParameter[[U]] = 2>
      class $Class[[Foo]] {
        void $Method[[f]]() {
          for(int $LocalVariable[[I]] = 0;
            $LocalVariable[[I]] < $TemplateParameter[[U]];) {}
        }
      };

      $Class[[G]] $Variable[[L]];
      void $Function[[f]]() {
        $Class[[Foo]]<123> $LocalVariable[[F]];
        $Class[[GP]]<&$Variable[[L]]> $LocalVariable[[LL]];
        $Class[[GR]]<$Variable[[L]]> $LocalVariable[[LLL]];
      }
    )cpp",
      R"cpp(
      template<typename $TemplateParameter[[T]], 
        void (T::*$TemplateParameter[[method]])(int)>
      struct $Class[[G]] {
        void $Method[[foo]](
            $TemplateParameter[[T]] *$Parameter[[O]]) {
          ($Parameter[[O]]->*$TemplateParameter[[method]])(10);
        }
      };
      struct $Class[[F]] {
        void $Method[[f]](int);
      };
      template<void (*$TemplateParameter[[Func]])()>
      struct $Class[[A]] {
        void $Method[[f]]() {
          (*$TemplateParameter[[Func]])();
        }
      };

      void $Function[[foo]]() {
        $Class[[F]] $LocalVariable[[FF]];
        $Class[[G]]<$Class[[F]], &$Class[[F]]::$Method[[f]]> $LocalVariable[[GG]];
        $LocalVariable[[GG]].$Method[[foo]](&$LocalVariable[[FF]]);
        $Class[[A]]<$Function[[foo]]> $LocalVariable[[AA]];
    )cpp",
      // Tokens that share a source range but have conflicting Kinds are not
      // highlighted.
      R"cpp(
      #define $Macro[[DEF_MULTIPLE]](X) namespace X { class X { int X; }; }
      #define $Macro[[DEF_CLASS]](T) class T {};
      // Preamble ends.
      $Macro[[DEF_MULTIPLE]](XYZ);
      $Macro[[DEF_MULTIPLE]](XYZW);
      $Macro[[DEF_CLASS]]($Class[[A]])
      #define $Macro[[MACRO_CONCAT]](X, V, T) T foo##X = V
      #define $Macro[[DEF_VAR]](X, V) int X = V
      #define $Macro[[DEF_VAR_T]](T, X, V) T X = V
      #define $Macro[[DEF_VAR_REV]](V, X) DEF_VAR(X, V)
      #define $Macro[[CPY]](X) X
      #define $Macro[[DEF_VAR_TYPE]](X, Y) X Y
      #define $Macro[[SOME_NAME]] variable
      #define $Macro[[SOME_NAME_SET]] variable2 = 123
      #define $Macro[[INC_VAR]](X) X += 2
      void $Function[[foo]]() {
        $Macro[[DEF_VAR]]($LocalVariable[[X]],  123);
        $Macro[[DEF_VAR_REV]](908, $LocalVariable[[XY]]);
        int $Macro[[CPY]]( $LocalVariable[[XX]] );
        $Macro[[DEF_VAR_TYPE]]($Class[[A]], $LocalVariable[[AA]]);
        double $Macro[[SOME_NAME]];
        int $Macro[[SOME_NAME_SET]];
        $LocalVariable[[variable]] = 20.1;
        $Macro[[MACRO_CONCAT]](var, 2, float);
        $Macro[[DEF_VAR_T]]($Class[[A]], $Macro[[CPY]](
              $Macro[[CPY]]($LocalVariable[[Nested]])),
            $Macro[[CPY]]($Class[[A]]()));
        $Macro[[INC_VAR]]($LocalVariable[[variable]]);
      }
      void $Macro[[SOME_NAME]]();
      $Macro[[DEF_VAR]]($Variable[[XYZ]], 567);
      $Macro[[DEF_VAR_REV]](756, $Variable[[AB]]);

      #define $Macro[[CALL_FN]](F) F();
      #define $Macro[[DEF_FN]](F) void F ()
      $Macro[[DEF_FN]]($Function[[g]]) {
        $Macro[[CALL_FN]]($Function[[foo]]);
      }
    )cpp",
      R"cpp(
      #define $Macro[[fail]](expr) expr
      #define $Macro[[assert]](COND) if (!(COND)) { fail("assertion failed" #COND); }
      // Preamble ends.
      int $Variable[[x]];
      int $Variable[[y]];
      int $Function[[f]]();
      void $Function[[foo]]() {
        $Macro[[assert]]($Variable[[x]] != $Variable[[y]]);
        $Macro[[assert]]($Variable[[x]] != $Function[[f]]());
      }
    )cpp",
      // highlighting all macro references
      R"cpp(
      #ifndef $Macro[[name]]
      #define $Macro[[name]]
      #endif

      #define $Macro[[test]]
      #undef $Macro[[test]]
$InactiveCode[[]]      #ifdef $Macro[[test]]
$InactiveCode[[]]      #endif

$InactiveCode[[]]      #if defined($Macro[[test]])
$InactiveCode[[]]      #endif
    )cpp",
      R"cpp(
      struct $Class[[S]] {
        float $Field[[Value]];
        $Class[[S]] *$Field[[Next]];
      };
      $Class[[S]] $Variable[[Global]][2] = {$Class[[S]](), $Class[[S]]()};
      void $Function[[f]]($Class[[S]] $Parameter[[P]]) {
        int $LocalVariable[[A]][2] = {1,2};
        auto [$Variable[[B1]], $Variable[[B2]]] = $LocalVariable[[A]];
        auto [$Variable[[G1]], $Variable[[G2]]] = $Variable[[Global]];
        $Class[[auto]] [$Variable[[P1]], $Variable[[P2]]] = $Parameter[[P]];
        // Highlights references to BindingDecls.
        $Variable[[B1]]++;
      }
    )cpp",
      R"cpp(
      template<class $TemplateParameter[[T]]>
      class $Class[[A]] {
        using $TemplateParameter[[TemplateParam1]] = $TemplateParameter[[T]];
        typedef $TemplateParameter[[T]] $TemplateParameter[[TemplateParam2]];
        using $Primitive[[IntType]] = int;

        using $Typedef[[Pointer]] = $TemplateParameter[[T]] *;
        using $Typedef[[LVReference]] = $TemplateParameter[[T]] &;
        using $Typedef[[RVReference]] = $TemplateParameter[[T]]&&;
        using $Typedef[[Array]] = $TemplateParameter[[T]]*[3];
        using $Typedef[[MemberPointer]] = int (A::*)(int);

        // Use various previously defined typedefs in a function type.
        void $Method[[func]](
          $Typedef[[Pointer]], $Typedef[[LVReference]], $Typedef[[RVReference]],
          $Typedef[[Array]], $Typedef[[MemberPointer]]);
      };
    )cpp",
      R"cpp(
      template <class $TemplateParameter[[T]]>
      void $Function[[phase1]]($TemplateParameter[[T]]);
      template <class $TemplateParameter[[T]]>
      void $Function[[foo]]($TemplateParameter[[T]] $Parameter[[P]]) {
        $Function[[phase1]]($Parameter[[P]]);
        $DependentName[[phase2]]($Parameter[[P]]);
      }
    )cpp",
      R"cpp(
      class $Class[[A]] {
        template <class $TemplateParameter[[T]]>
        void $Method[[bar]]($TemplateParameter[[T]]);
      };

      template <class $TemplateParameter[[U]]>
      void $Function[[foo]]($TemplateParameter[[U]] $Parameter[[P]]) {
        $Class[[A]]().$Method[[bar]]($Parameter[[P]]);
      }
    )cpp",
      R"cpp(
      struct $Class[[A]] {
        template <class $TemplateParameter[[T]]>
        static void $StaticMethod[[foo]]($TemplateParameter[[T]]);
      };

      template <class $TemplateParameter[[T]]>
      struct $Class[[B]] {
        void $Method[[bar]]() {
          $Class[[A]]::$StaticMethod[[foo]]($TemplateParameter[[T]]());
        }
      };
    )cpp",
      R"cpp(
      template <class $TemplateParameter[[T]]>
      void $Function[[foo]](typename $TemplateParameter[[T]]::$DependentType[[Type]]
                                            = $TemplateParameter[[T]]::$DependentName[[val]]);
    )cpp",
      R"cpp(
      template <class $TemplateParameter[[T]]>
      void $Function[[foo]]($TemplateParameter[[T]] $Parameter[[P]]) {
        $Parameter[[P]].$DependentName[[Field]];
      }
    )cpp",
      R"cpp(
      template <class $TemplateParameter[[T]]>
      class $Class[[A]] {
        int $Method[[foo]]() {
          return $TemplateParameter[[T]]::$DependentName[[Field]];
        }
      };
    )cpp",
      // Highlighting the using decl as the underlying using shadow decl.
      R"cpp(
      void $Function[[foo]]();
      using ::$Function[[foo]];
    )cpp",
      // Highlighting of template template arguments.
      R"cpp(
      template <template <class> class $TemplateParameter[[TT]],
                template <class> class ...$TemplateParameter[[TTs]]>
      struct $Class[[Foo]] {
        $Class[[Foo]]<$TemplateParameter[[TT]], $TemplateParameter[[TTs]]...>
          *$Field[[t]];
      }
    )cpp",
      // Inactive code highlighting
      R"cpp(
      // Code in the preamble.
      // Inactive lines get an empty InactiveCode token at the beginning.
$InactiveCode[[]]      #ifdef $Macro[[test]]
$InactiveCode[[]]      #endif

      // A declaration to cause the preamble to end.
      int $Variable[[EndPreamble]];

      // Code after the preamble.
      // Code inside inactive blocks does not get regular highlightings
      // because it's not part of the AST.
$InactiveCode[[]]      #ifdef $Macro[[test]]
$InactiveCode[[]]      int Inactive2;
$InactiveCode[[]]      #endif

      #ifndef $Macro[[test]]
      int $Variable[[Active1]];
      #endif

$InactiveCode[[]]      #ifdef $Macro[[test]]
$InactiveCode[[]]      int Inactive3;
$InactiveCode[[]]      #else
      int $Variable[[Active2]];
      #endif
    )cpp"};
  for (const auto &TestCase : TestCases) {
    checkHighlightings(TestCase);
  }

  checkHighlightings(R"cpp(
    class $Class[[A]] {
      #include "imp.h"
    };
    #endif
  )cpp",
                     {{"imp.h", R"cpp(
    int someMethod();
    void otherMethod();
  )cpp"}});

  // A separate test for macros in headers.
  checkHighlightings(R"cpp(
    #include "imp.h"
    $Macro[[DEFINE_Y]]
    $Macro[[DXYZ_Y]](A);
  )cpp",
                     {{"imp.h", R"cpp(
    #define DXYZ(X) class X {};
    #define DXYZ_Y(Y) DXYZ(x##Y)
    #define DEFINE(X) int X;
    #define DEFINE_Y DEFINE(Y)
  )cpp"}});
}

TEST(SemanticHighlighting, GeneratesHighlightsWhenFileChange) {
  class HighlightingsCounterDiagConsumer : public DiagnosticsConsumer {
  public:
    std::atomic<int> Count = {0};

    void onDiagnosticsReady(PathRef, std::vector<Diag>) override {}
    void onHighlightingsReady(
        PathRef File, std::vector<HighlightingToken> Highlightings) override {
      ++Count;
    }
  };

  auto FooCpp = testPath("foo.cpp");
  MockFSProvider FS;
  FS.Files[FooCpp] = "";

  MockCompilationDatabase MCD;
  HighlightingsCounterDiagConsumer DiagConsumer;
  ClangdServer Server(MCD, FS, DiagConsumer, ClangdServer::optsForTest());
  Server.addDocument(FooCpp, "int a;");
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for server";
  ASSERT_EQ(DiagConsumer.Count, 1);
}

TEST(SemanticHighlighting, toSemanticHighlightingInformation) {
  auto CreatePosition = [](int Line, int Character) -> Position {
    Position Pos;
    Pos.line = Line;
    Pos.character = Character;
    return Pos;
  };

  std::vector<LineHighlightings> Tokens{
      {3,
       {{HighlightingKind::Variable,
         Range{CreatePosition(3, 8), CreatePosition(3, 12)}},
        {HighlightingKind::Function,
         Range{CreatePosition(3, 4), CreatePosition(3, 7)}}},
       /* IsInactive = */ false},
      {1,
       {{HighlightingKind::Variable,
         Range{CreatePosition(1, 1), CreatePosition(1, 5)}}},
       /* IsInactive = */ true}};
  std::vector<SemanticHighlightingInformation> ActualResults =
      toSemanticHighlightingInformation(Tokens);
  std::vector<SemanticHighlightingInformation> ExpectedResults = {
      {3, "AAAACAAEAAAAAAAEAAMAAw=="}, {1, "AAAAAQAEAAA="}};
  EXPECT_EQ(ActualResults, ExpectedResults);
}

TEST(SemanticHighlighting, HighlightingDiffer) {
  struct {
    llvm::StringRef OldCode;
    llvm::StringRef NewCode;
  } TestCases[]{{
                    R"(
        $Variable[[A]]
        $Class[[B]]
        $Function[[C]]
      )",
                    R"(
        $Variable[[A]]
        $Class[[D]]
        $Function[[C]]
      )"},
                {
                    R"(
        $Class[[C]]
        $Field[[F]]
        $Variable[[V]]
        $Class[[C]] $Variable[[V]] $Field[[F]]
      )",
                    R"(
        $Class[[C]]
        $Field[[F]]
       ^$Function[[F]]
        $Class[[C]] $Variable[[V]] $Field[[F]]
      )"},
                {
                    R"(

        $Class[[A]]
        $Variable[[A]]
      )",
                    R"(

       ^
       ^$Class[[A]]
       ^$Variable[[A]]
      )"},
                {
                    R"(
        $Class[[C]]
        $Field[[F]]
        $Variable[[V]]
        $Class[[C]] $Variable[[V]] $Field[[F]]
      )",
                    R"(
        $Class[[C]]
       ^
       ^
        $Class[[C]] $Variable[[V]] $Field[[F]]
      )"},
                {
                    R"(
        $Class[[A]]
        $Variable[[A]]
        $Variable[[A]]
      )",
                    R"(
        $Class[[A]]
       ^$Variable[[AA]]
        $Variable[[A]]
      )"},
                {
                    R"(
        $Class[[A]]
        $Variable[[A]]
      )",
                    R"(
        $Class[[A]]
        $Variable[[A]]
       ^$Class[[A]]
       ^$Variable[[A]]
      )"},
                {
                    R"(
        $Variable[[A]]
        $Variable[[A]]
        $Variable[[A]]
      )",
                    R"(
       ^$Class[[A]]
       ^$Class[[A]]
       ^$Class[[A]]
      )"}};

  for (const auto &Test : TestCases)
    checkDiffedHighlights(Test.OldCode, Test.NewCode);
}

TEST(SemanticHighlighting, DiffBeyondTheEndOfFile) {
  llvm::StringRef OldCode =
      R"(
        $Class[[A]]
        $Variable[[A]]
        $Class[[A]]
        $Variable[[A]]
      )";
  llvm::StringRef NewCode =
      R"(
        $Class[[A]] // line 1
        $Variable[[A]] // line 2
      )";

  Annotations OldTest(OldCode);
  Annotations NewTest(NewCode);
  std::vector<HighlightingToken> OldTokens = getExpectedTokens(OldTest);
  std::vector<HighlightingToken> NewTokens = getExpectedTokens(NewTest);

  auto ActualDiff = diffHighlightings(NewTokens, OldTokens);
  EXPECT_THAT(ActualDiff,
              testing::UnorderedElementsAre(
                  testing::AllOf(LineNumber(3), EmptyHighlightings()),
                  testing::AllOf(LineNumber(4), EmptyHighlightings())));
}

} // namespace
} // namespace clangd
} // namespace clang
