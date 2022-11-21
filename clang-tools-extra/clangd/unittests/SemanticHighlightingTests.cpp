//==- SemanticHighlightingTests.cpp - SemanticHighlighting tests-*- C++ -* -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include <algorithm>

namespace clang {
namespace clangd {
namespace {

using testing::IsEmpty;
using testing::SizeIs;

/// Annotates the input code with provided semantic highlightings. Results look
/// something like:
///   class $Class[[X]] {
///     $Primitive[[int]] $Field[[a]] = 0;
///   };
std::string annotate(llvm::StringRef Input,
                     llvm::ArrayRef<HighlightingToken> Tokens) {
  assert(llvm::is_sorted(
      Tokens, [](const HighlightingToken &L, const HighlightingToken &R) {
        return L.R.start < R.R.start;
      }));

  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  unsigned NextChar = 0;
  for (auto &T : Tokens) {
    unsigned StartOffset = llvm::cantFail(positionToOffset(Input, T.R.start));
    unsigned EndOffset = llvm::cantFail(positionToOffset(Input, T.R.end));
    assert(StartOffset <= EndOffset);
    assert(NextChar <= StartOffset);

    bool hasDef =
        T.Modifiers & (1 << uint32_t(HighlightingModifier::Definition));
    bool hasDecl =
        T.Modifiers & (1 << uint32_t(HighlightingModifier::Declaration));
    EXPECT_TRUE(!hasDef || hasDecl);

    OS << Input.substr(NextChar, StartOffset - NextChar);
    OS << '$' << T.Kind;
    for (unsigned I = 0;
         I <= static_cast<uint32_t>(HighlightingModifier::LastModifier); ++I) {
      if (T.Modifiers & (1 << I)) {
        // _decl_def is common and redundant, just print _def instead.
        if (I != uint32_t(HighlightingModifier::Declaration) || !hasDef)
          OS << '_' << static_cast<HighlightingModifier>(I);
      }
    }
    OS << "[[" << Input.substr(StartOffset, EndOffset - StartOffset) << "]]";
    NextChar = EndOffset;
  }
  OS << Input.substr(NextChar);
  return std::move(OS.str());
}

void checkHighlightings(llvm::StringRef Code,
                        std::vector<std::pair</*FileName*/ llvm::StringRef,
                                              /*FileContent*/ llvm::StringRef>>
                            AdditionalFiles = {},
                        uint32_t ModifierMask = -1,
                        std::vector<std::string> AdditionalArgs = {}) {
  Annotations Test(Code);
  TestTU TU;
  TU.Code = std::string(Test.code());

  TU.ExtraArgs.push_back("-std=c++20");
  TU.ExtraArgs.push_back("-xobjective-c++");
  TU.ExtraArgs.insert(std::end(TU.ExtraArgs), std::begin(AdditionalArgs),
                      std::end(AdditionalArgs));

  for (auto File : AdditionalFiles)
    TU.AdditionalFiles.insert({File.first, std::string(File.second)});
  auto AST = TU.build();
  auto Actual = getSemanticHighlightings(AST);
  for (auto &Token : Actual)
    Token.Modifiers &= ModifierMask;

  EXPECT_EQ(Code, annotate(Test.code(), Actual));
}

constexpr static uint32_t ScopeModifierMask =
    1 << unsigned(HighlightingModifier::FunctionScope) |
    1 << unsigned(HighlightingModifier::ClassScope) |
    1 << unsigned(HighlightingModifier::FileScope) |
    1 << unsigned(HighlightingModifier::GlobalScope);

TEST(SemanticHighlighting, GetsCorrectTokens) {
  const char *TestCases[] = {
      R"cpp(
      struct $Class_def[[AS]] {
        double $Field_decl[[SomeMember]];
      };
      struct {
      } $Variable_def[[S]];
      void $Function_def[[foo]](int $Parameter_def[[A]], $Class[[AS]] $Parameter_def[[As]]) {
        $Primitive_deduced_defaultLibrary[[auto]] $LocalVariable_def[[VeryLongVariableName]] = 12312;
        $Class[[AS]]     $LocalVariable_def[[AA]];
        $Primitive_deduced_defaultLibrary[[auto]] $LocalVariable_def[[L]] = $LocalVariable[[AA]].$Field[[SomeMember]] + $Parameter[[A]];
        auto $LocalVariable_def[[FN]] = [ $LocalVariable[[AA]]](int $Parameter_def[[A]]) -> void {};
        $LocalVariable[[FN]](12312);
      }
    )cpp",
      R"cpp(
      void $Function_decl[[foo]](int);
      void $Function_decl[[Gah]]();
      void $Function_def[[foo]]() {
        auto $LocalVariable_def[[Bou]] = $Function[[Gah]];
      }
      struct $Class_def[[A]] {
        void $Method_decl[[abc]]();
      };
    )cpp",
      R"cpp(
      namespace $Namespace_decl[[abc]] {
        template<typename $TemplateParameter_def[[T]]>
        struct $Class_def[[A]] {
          $TemplateParameter[[T]] $Field_decl[[t]];
        };
      }
      template<typename $TemplateParameter_def[[T]]>
      struct $Class_def[[C]] : $Namespace[[abc]]::$Class[[A]]<$TemplateParameter[[T]]> {
        typename $TemplateParameter[[T]]::$Type_dependentName[[A]]* $Field_decl[[D]];
      };
      $Namespace[[abc]]::$Class[[A]]<int> $Variable_def[[AA]];
      typedef $Namespace[[abc]]::$Class[[A]]<int> $Class_decl[[AAA]];
      struct $Class_def[[B]] {
        $Class_decl_constrDestr[[B]]();
        ~$Class_decl_constrDestr[[B]]();
        void operator<<($Class[[B]]);
        $Class[[AAA]] $Field_decl[[AA]];
      };
      $Class[[B]]::$Class_def_constrDestr[[B]]() {}
      $Class[[B]]::~$Class_def_constrDestr[[B]]() {}
      void $Function_def[[f]] () {
        $Class[[B]] $LocalVariable_def[[BB]] = $Class[[B]]();
        $LocalVariable[[BB]].~$Class_constrDestr[[B]]();
        $Class[[B]]();
      }
    )cpp",
      R"cpp(
      enum class $Enum_decl[[E]] {
        $EnumConstant_decl_readonly[[A]],
        $EnumConstant_decl_readonly[[B]],
      };
      enum $Enum_decl[[EE]] {
        $EnumConstant_decl_readonly[[Hi]],
      };
      struct $Class_def[[A]] {
        $Enum[[E]] $Field_decl[[EEE]];
        $Enum[[EE]] $Field_decl[[EEEE]];
      };
      int $Variable_def[[I]] = $EnumConstant_readonly[[Hi]];
      $Enum[[E]] $Variable_def[[L]] = $Enum[[E]]::$EnumConstant_readonly[[B]];
    )cpp",
      R"cpp(
      namespace $Namespace_decl[[abc]] {
        namespace {}
        namespace $Namespace_decl[[bcd]] {
          struct $Class_def[[A]] {};
          namespace $Namespace_decl[[cde]] {
            struct $Class_def[[A]] {
              enum class $Enum_decl[[B]] {
                $EnumConstant_decl_readonly[[Hi]],
              };
            };
          }
        }
      }
      using namespace $Namespace[[abc]]::$Namespace[[bcd]];
      namespace $Namespace_decl[[vwz]] =
            $Namespace[[abc]]::$Namespace[[bcd]]::$Namespace[[cde]];
      $Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable_def[[AA]];
      $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]] $Variable_def[[AAA]] =
            $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]]::$EnumConstant_readonly[[Hi]];
      ::$Namespace[[vwz]]::$Class[[A]] $Variable_def[[B]];
      ::$Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable_def[[BB]];
    )cpp",
      R"cpp(
      struct $Class_def[[D]] {
        double $Field_decl[[C]];
      };
      struct $Class_def[[A]] {
        double $Field_decl[[B]];
        $Class[[D]] $Field_decl[[E]];
        static double $StaticField_decl_static[[S]];
        static void $StaticMethod_def_static[[bar]]() {}
        void $Method_def[[foo]]() {
          $Field[[B]] = 123;
          this->$Field[[B]] = 156;
          this->$Method[[foo]]();
          $Method[[foo]]();
          $StaticMethod_static[[bar]]();
          $StaticField_static[[S]] = 90.1;
        }
      };
      void $Function_def[[foo]]() {
        $Class[[A]] $LocalVariable_def[[AA]];
        $LocalVariable[[AA]].$Field[[B]] += 2;
        $LocalVariable[[AA]].$Method[[foo]]();
        $LocalVariable[[AA]].$Field[[E]].$Field[[C]];
        $Class[[A]]::$StaticField_static[[S]] = 90;
      }
    )cpp",
      R"cpp(
      struct $Class_def[[AA]] {
        int $Field_decl[[A]];
      };
      int $Variable_def[[B]];
      $Class[[AA]] $Variable_def[[A]]{$Variable[[B]]};
    )cpp",
      R"cpp(
      namespace $Namespace_decl[[a]] {
        struct $Class_def[[A]] {};
        typedef char $Primitive_decl[[C]];
      }
      typedef $Namespace[[a]]::$Class[[A]] $Class_decl[[B]];
      using $Class_decl[[BB]] = $Namespace[[a]]::$Class[[A]];
      enum class $Enum_decl[[E]] {};
      typedef $Enum[[E]] $Enum_decl[[C]];
      typedef $Enum[[C]] $Enum_decl[[CC]];
      using $Enum_decl[[CD]] = $Enum[[CC]];
      $Enum[[CC]] $Function_decl[[f]]($Class[[B]]);
      $Enum[[CD]] $Function_decl[[f]]($Class[[BB]]);
      typedef $Namespace[[a]]::$Primitive[[C]] $Primitive_decl[[PC]];
      typedef float $Primitive_decl[[F]];
    )cpp",
      R"cpp(
      template<typename $TemplateParameter_def[[T]], typename = void>
      class $Class_def[[A]] {
        $TemplateParameter[[T]] $Field_decl[[AA]];
        $TemplateParameter[[T]] $Method_decl[[foo]]();
      };
      template<class $TemplateParameter_def[[TT]]>
      class $Class_def[[B]] {
        $Class[[A]]<$TemplateParameter[[TT]]> $Field_decl[[AA]];
      };
      template<class $TemplateParameter_def[[TT]], class $TemplateParameter_def[[GG]]>
      class $Class_def[[BB]] {};
      template<class $TemplateParameter_def[[T]]>
      class $Class_def[[BB]]<$TemplateParameter[[T]], int> {};
      template<class $TemplateParameter_def[[T]]>
      class $Class_def[[BB]]<$TemplateParameter[[T]], $TemplateParameter[[T]]*> {};

      template<template<class> class $TemplateParameter_def[[T]], class $TemplateParameter_def[[C]]>
      $TemplateParameter[[T]]<$TemplateParameter[[C]]> $Function_decl[[f]]();

      template<typename>
      class $Class_def[[Foo]] {};

      template<typename $TemplateParameter_def[[T]]>
      void $Function_decl[[foo]]($TemplateParameter[[T]] ...);
    )cpp",
      R"cpp(
      template <class $TemplateParameter_def[[T]]>
      struct $Class_def[[Tmpl]] {$TemplateParameter[[T]] $Field_decl[[x]] = 0;};
      extern template struct $Class_def[[Tmpl]]<float>;
      template struct $Class_def[[Tmpl]]<double>;
    )cpp",
      // This test is to guard against highlightings disappearing when using
      // conversion operators as their behaviour in the clang AST differ from
      // other CXXMethodDecls.
      R"cpp(
      class $Class_def[[Foo]] {};
      struct $Class_def[[Bar]] {
        explicit operator $Class[[Foo]]*() const;
        explicit operator int() const;
        operator $Class[[Foo]]();
      };
      void $Function_def[[f]]() {
        $Class[[Bar]] $LocalVariable_def[[B]];
        $Class[[Foo]] $LocalVariable_def[[F]] = $LocalVariable[[B]];
        $Class[[Foo]] *$LocalVariable_def[[FP]] = ($Class[[Foo]]*)$LocalVariable[[B]];
        int $LocalVariable_def[[I]] = (int)$LocalVariable[[B]];
      }
    )cpp",
      R"cpp(
      struct $Class_def[[B]] {};
      struct $Class_def[[A]] {
        $Class[[B]] $Field_decl[[BB]];
        $Class[[A]] &operator=($Class[[A]] &&$Parameter_def[[O]]);
      };

      $Class[[A]] &$Class[[A]]::operator=($Class[[A]] &&$Parameter_def[[O]]) = default;
    )cpp",
      R"cpp(
      enum $Enum_decl[[En]] {
        $EnumConstant_decl_readonly[[EC]],
      };
      class $Class_def[[Foo]] {};
      class $Class_def[[Bar]] {
      public:
        $Class[[Foo]] $Field_decl[[Fo]];
        $Enum[[En]] $Field_decl[[E]];
        int $Field_decl[[I]];
        $Class_def_constrDestr[[Bar]] ($Class[[Foo]] $Parameter_def[[F]],
                $Enum[[En]] $Parameter_def[[E]])
        : $Field[[Fo]] ($Parameter[[F]]), $Field[[E]] ($Parameter[[E]]),
          $Field[[I]] (123) {}
      };
      class $Class_def[[Bar2]] : public $Class[[Bar]] {
        $Class_def_constrDestr[[Bar2]]() : $Class[[Bar]]($Class[[Foo]](), $EnumConstant_readonly[[EC]]) {}
      };
    )cpp",
      R"cpp(
      enum $Enum_decl[[E]] {
        $EnumConstant_decl_readonly[[E]],
      };
      class $Class_def[[Foo]] {};
      $Enum_deduced[[auto]] $Variable_def[[AE]] = $Enum[[E]]::$EnumConstant_readonly[[E]];
      $Class_deduced[[auto]] $Variable_def[[AF]] = $Class[[Foo]]();
      $Class_deduced[[decltype]](auto) $Variable_def[[AF2]] = $Class[[Foo]]();
      $Class_deduced[[auto]] *$Variable_def[[AFP]] = &$Variable[[AF]];
      $Enum_deduced[[auto]] &$Variable_def[[AER]] = $Variable[[AE]];
      $Primitive_deduced_defaultLibrary[[auto]] $Variable_def[[Form]] = 10.2 + 2 * 4;
      $Primitive_deduced_defaultLibrary[[decltype]]($Variable[[Form]]) $Variable_def[[F]] = 10;
      auto $Variable_def[[Fun]] = []()->void{};
    )cpp",
      R"cpp(
      class $Class_def[[G]] {};
      template<$Class[[G]] *$TemplateParameter_def_readonly[[U]]>
      class $Class_def[[GP]] {};
      template<$Class[[G]] &$TemplateParameter_def_readonly[[U]]>
      class $Class_def[[GR]] {};
      template<int *$TemplateParameter_def_readonly[[U]]>
      class $Class_def[[IP]] {
        void $Method_def[[f]]() {
          *$TemplateParameter_readonly[[U]] += 5;
        }
      };
      template<unsigned $TemplateParameter_def_readonly[[U]] = 2>
      class $Class_def[[Foo]] {
        void $Method_def[[f]]() {
          for(int $LocalVariable_def[[I]] = 0;
            $LocalVariable[[I]] < $TemplateParameter_readonly[[U]];) {}
        }
      };

      $Class[[G]] $Variable_def[[L]];
      void $Function_def[[f]]() {
        $Class[[Foo]]<123> $LocalVariable_def[[F]];
        $Class[[GP]]<&$Variable[[L]]> $LocalVariable_def[[LL]];
        $Class[[GR]]<$Variable[[L]]> $LocalVariable_def[[LLL]];
      }
    )cpp",
      R"cpp(
      template<typename $TemplateParameter_def[[T]],
        void ($TemplateParameter[[T]]::*$TemplateParameter_def_readonly[[method]])(int)>
      struct $Class_def[[G]] {
        void $Method_def[[foo]](
            $TemplateParameter[[T]] *$Parameter_def[[O]]) {
          ($Parameter[[O]]->*$TemplateParameter_readonly[[method]])(10);
        }
      };
      struct $Class_def[[F]] {
        void $Method_decl[[f]](int);
      };
      template<void (*$TemplateParameter_def_readonly[[Func]])()>
      struct $Class_def[[A]] {
        void $Method_def[[f]]() {
          (*$TemplateParameter_readonly[[Func]])();
        }
      };

      void $Function_def[[foo]]() {
        $Class[[F]] $LocalVariable_def[[FF]];
        $Class[[G]]<$Class[[F]], &$Class[[F]]::$Method[[f]]> $LocalVariable_def[[GG]];
        $LocalVariable[[GG]].$Method[[foo]](&$LocalVariable_usedAsMutablePointer[[FF]]);
        $Class[[A]]<$Function[[foo]]> $LocalVariable_def[[AA]];
      }
    )cpp",
      // Tokens that share a source range but have conflicting Kinds are not
      // highlighted.
      R"cpp(
      #define $Macro_decl[[DEF_MULTIPLE]](X) namespace X { class X { int X; }; }
      #define $Macro_decl[[DEF_CLASS]](T) class T {};
      // Preamble ends.
      $Macro[[DEF_MULTIPLE]](XYZ);
      $Macro[[DEF_MULTIPLE]](XYZW);
      $Macro[[DEF_CLASS]]($Class_def[[A]])
      #define $Macro_decl[[MACRO_CONCAT]](X, V, T) T foo##X = V
      #define $Macro_decl[[DEF_VAR]](X, V) int X = V
      #define $Macro_decl[[DEF_VAR_T]](T, X, V) T X = V
      #define $Macro_decl[[DEF_VAR_REV]](V, X) DEF_VAR(X, V)
      #define $Macro_decl[[CPY]](X) X
      #define $Macro_decl[[DEF_VAR_TYPE]](X, Y) X Y
      #define $Macro_decl[[SOME_NAME]] variable
      #define $Macro_decl[[SOME_NAME_SET]] variable2 = 123
      #define $Macro_decl[[INC_VAR]](X) X += 2
      void $Function_def[[foo]]() {
        $Macro[[DEF_VAR]]($LocalVariable_def[[X]],  123);
        $Macro[[DEF_VAR_REV]](908, $LocalVariable_def[[XY]]);
        int $Macro[[CPY]]( $LocalVariable_def[[XX]] );
        $Macro[[DEF_VAR_TYPE]]($Class[[A]], $LocalVariable_def[[AA]]);
        double $Macro[[SOME_NAME]];
        int $Macro[[SOME_NAME_SET]];
        $LocalVariable[[variable]] = 20.1;
        $Macro[[MACRO_CONCAT]](var, 2, float);
        $Macro[[DEF_VAR_T]]($Class[[A]], $Macro[[CPY]](
              $Macro[[CPY]]($LocalVariable_def[[Nested]])),
            $Macro[[CPY]]($Class[[A]]()));
        $Macro[[INC_VAR]]($LocalVariable[[variable]]);
      }
      void $Macro[[SOME_NAME]]();
      $Macro[[DEF_VAR]]($Variable_def[[MMMMM]], 567);
      $Macro[[DEF_VAR_REV]](756, $Variable_def[[AB]]);

      #define $Macro_decl[[CALL_FN]](F) F();
      #define $Macro_decl[[DEF_FN]](F) void F ()
      $Macro[[DEF_FN]]($Function_def[[g]]) {
        $Macro[[CALL_FN]]($Function[[foo]]);
      }
    )cpp",
      R"cpp(
      #define $Macro_decl[[fail]](expr) expr
      #define $Macro_decl[[assert]](COND) if (!(COND)) { fail("assertion failed" #COND); }
      // Preamble ends.
      int $Variable_def[[x]];
      int $Variable_def[[y]];
      int $Function_decl[[f]]();
      void $Function_def[[foo]]() {
        $Macro[[assert]]($Variable[[x]] != $Variable[[y]]);
        $Macro[[assert]]($Variable[[x]] != $Function[[f]]());
      }
    )cpp",
      // highlighting all macro references
      R"cpp(
      #ifndef $Macro[[name]]
      #define $Macro_decl[[name]]
      #endif

      #define $Macro_decl[[test]]
      #undef $Macro[[test]]
$InactiveCode[[#ifdef test]]
$InactiveCode[[#endif]]

$InactiveCode[[#if defined(test)]]
$InactiveCode[[#endif]]
    )cpp",
      R"cpp(
      struct $Class_def[[S]] {
        float $Field_decl[[Value]];
        $Class[[S]] *$Field_decl[[Next]];
      };
      $Class[[S]] $Variable_def[[Global]][2] = {$Class[[S]](), $Class[[S]]()};
      auto [$Variable_decl[[G1]], $Variable_decl[[G2]]] = $Variable[[Global]];
      void $Function_def[[f]]($Class[[S]] $Parameter_def[[P]]) {
        int $LocalVariable_def[[A]][2] = {1,2};
        auto [$LocalVariable_decl[[B1]], $LocalVariable_decl[[B2]]] = $LocalVariable[[A]];
        auto [$LocalVariable_decl[[G1]], $LocalVariable_decl[[G2]]] = $Variable[[Global]];
        $Class_deduced[[auto]] [$LocalVariable_decl[[P1]], $LocalVariable_decl[[P2]]] = $Parameter[[P]];
        // Highlights references to BindingDecls.
        $LocalVariable[[B1]]++;
      }
    )cpp",
      R"cpp(
      template<class $TemplateParameter_def[[T]]>
      class $Class_def[[A]] {
        using $TemplateParameter_decl[[TemplateParam1]] = $TemplateParameter[[T]];
        typedef $TemplateParameter[[T]] $TemplateParameter_decl[[TemplateParam2]];
        using $Primitive_decl[[IntType]] = int;

        using $Typedef_decl[[Pointer]] = $TemplateParameter[[T]] *;
        using $Typedef_decl[[LVReference]] = $TemplateParameter[[T]] &;
        using $Typedef_decl[[RVReference]] = $TemplateParameter[[T]]&&;
        using $Typedef_decl[[Array]] = $TemplateParameter[[T]]*[3];
        using $Typedef_decl[[MemberPointer]] = int ($Class[[A]]::*)(int);

        // Use various previously defined typedefs in a function type.
        void $Method_decl[[func]](
          $Typedef[[Pointer]], $Typedef[[LVReference]], $Typedef[[RVReference]],
          $Typedef[[Array]], $Typedef[[MemberPointer]]);
      };
    )cpp",
      R"cpp(
      template <class $TemplateParameter_def[[T]]>
      void $Function_decl[[phase1]]($TemplateParameter[[T]]);
      template <class $TemplateParameter_def[[T]]>
      void $Function_def[[foo]]($TemplateParameter[[T]] $Parameter_def[[P]]) {
        $Function[[phase1]]($Parameter[[P]]);
        $Unknown_dependentName[[phase2]]($Parameter[[P]]);
      }
    )cpp",
      R"cpp(
      class $Class_def[[A]] {
        template <class $TemplateParameter_def[[T]]>
        void $Method_decl[[bar]]($TemplateParameter[[T]]);
      };

      template <class $TemplateParameter_def[[U]]>
      void $Function_def[[foo]]($TemplateParameter[[U]] $Parameter_def[[P]]) {
        $Class[[A]]().$Method[[bar]]($Parameter[[P]]);
      }
    )cpp",
      R"cpp(
      struct $Class_def[[A]] {
        template <class $TemplateParameter_def[[T]]>
        static void $StaticMethod_decl_static[[foo]]($TemplateParameter[[T]]);
      };

      template <class $TemplateParameter_def[[T]]>
      struct $Class_def[[B]] {
        void $Method_def[[bar]]() {
          $Class[[A]]::$StaticMethod_static[[foo]]($TemplateParameter[[T]]());
        }
      };
    )cpp",
      R"cpp(
      template <class $TemplateParameter_def[[T]]>
      void $Function_decl[[foo]](typename $TemplateParameter[[T]]::$Type_dependentName[[Type]]
                                            = $TemplateParameter[[T]]::$Unknown_dependentName[[val]]);
    )cpp",
      R"cpp(
      template <class $TemplateParameter_def[[T]]>
      void $Function_def[[foo]]($TemplateParameter[[T]] $Parameter_def[[P]]) {
        $Parameter[[P]].$Unknown_dependentName[[Field]];
      }
    )cpp",
      R"cpp(
      template <class $TemplateParameter_def[[T]]>
      class $Class_def[[A]] {
        int $Method_def[[foo]]() {
          return $TemplateParameter[[T]]::$Unknown_dependentName[[Field]];
        }
      };
    )cpp",
      // Highlighting the using decl as the underlying using shadow decl.
      R"cpp(
      void $Function_decl[[foo]]();
      using ::$Function[[foo]];
    )cpp",
      // Highlighting of template template arguments.
      R"cpp(
      template <template <class> class $TemplateParameter_def[[TT]],
                template <class> class ...$TemplateParameter_def[[TTs]]>
      struct $Class_def[[Foo]] {
        $Class[[Foo]]<$TemplateParameter[[TT]], $TemplateParameter[[TTs]]...>
          *$Field_decl[[t]];
      };
    )cpp",
      // Inactive code highlighting
      R"cpp(
      // Code in the preamble.
      // Inactive lines get an empty InactiveCode token at the beginning.
$InactiveCode[[#ifdef test]]
$InactiveCode[[#endif]]

      // A declaration to cause the preamble to end.
      int $Variable_def[[EndPreamble]];

      // Code after the preamble.
      // Code inside inactive blocks does not get regular highlightings
      // because it's not part of the AST.
      #define $Macro_decl[[test2]]
$InactiveCode[[#if defined(test)]]
$InactiveCode[[int Inactive2;]]
$InactiveCode[[#elif defined(test2)]]
      int $Variable_def[[Active1]];
$InactiveCode[[#else]]
$InactiveCode[[int Inactive3;]]
$InactiveCode[[#endif]]

      #ifndef $Macro[[test]]
      int $Variable_def[[Active2]];
      #endif

$InactiveCode[[#ifdef test]]
$InactiveCode[[int Inactive4;]]
$InactiveCode[[#else]]
      int $Variable_def[[Active3]];
      #endif
    )cpp",
      // Argument to 'sizeof...'
      R"cpp(
      template <typename... $TemplateParameter_def[[Elements]]>
      struct $Class_def[[TupleSize]] {
        static const int $StaticField_decl_readonly_static[[size]] =
sizeof...($TemplateParameter[[Elements]]);
      };
    )cpp",
      // More dependent types
      R"cpp(
      template <typename $TemplateParameter_def[[T]]>
      struct $Class_def[[Waldo]] {
        using $Typedef_decl[[Location1]] = typename $TemplateParameter[[T]]
            ::$Type_dependentName[[Resolver]]::$Type_dependentName[[Location]];
        using $Typedef_decl[[Location2]] = typename $TemplateParameter[[T]]
            ::template $Type_dependentName[[Resolver]]<$TemplateParameter[[T]]>
            ::$Type_dependentName[[Location]];
        using $Typedef_decl[[Location3]] = typename $TemplateParameter[[T]]
            ::$Type_dependentName[[Resolver]]
            ::template $Type_dependentName[[Location]]<$TemplateParameter[[T]]>;
        static const int $StaticField_decl_readonly_static[[Value]] = $TemplateParameter[[T]]
            ::$Type_dependentName[[Resolver]]::$Unknown_dependentName[[Value]];
      };
    )cpp",
      // Dependent name with heuristic target
      R"cpp(
      template <typename>
      struct $Class_def[[Foo]] {
        int $Field_decl[[Waldo]];
        void $Method_def[[bar]]() {
          $Class[[Foo]]().$Field_dependentName[[Waldo]];
        }
        template <typename $TemplateParameter_def[[U]]>
        void $Method_def[[bar1]]() {
          $Class[[Foo]]<$TemplateParameter[[U]]>().$Field_dependentName[[Waldo]];
        }

        void $Method_decl[[Overload]]();
        void $Method_decl_readonly[[Overload]]() const;
      };
      template <typename $TemplateParameter_def[[T]]>
      void $Function_def[[baz]]($Class[[Foo]]<$TemplateParameter[[T]]> $Parameter_def[[o]]) {
        $Parameter[[o]].$Method_readonly_dependentName[[Overload]]();
      }
    )cpp",
      // Concepts
      R"cpp(
      template <typename $TemplateParameter_def[[T]]>
      concept $Concept_decl[[Fooable]] =
          requires($TemplateParameter[[T]] $Parameter_def[[F]]) {
            $Parameter[[F]].$Unknown_dependentName[[foo]]();
          };
      template <typename $TemplateParameter_def[[T]]>
          requires $Concept[[Fooable]]<$TemplateParameter[[T]]>
      void $Function_def[[bar]]($TemplateParameter[[T]] $Parameter_def[[F]]) {
        $Parameter[[F]].$Unknown_dependentName[[foo]]();
      }
    )cpp",
      // Dependent template name
      R"cpp(
      template <template <typename> class> struct $Class_def[[A]] {};
      template <typename $TemplateParameter_def[[T]]>
      using $Typedef_decl[[W]] = $Class[[A]]<
        $TemplateParameter[[T]]::template $Class_dependentName[[Waldo]]
      >;
    )cpp",
      R"cpp(
      class $Class_def_abstract[[Abstract]] {
      public:
        virtual void $Method_decl_abstract_virtual[[pure]]() = 0;
        virtual void $Method_decl_virtual[[impl]]();
      };
      void $Function_def[[foo]]($Class_abstract[[Abstract]]* $Parameter_def[[A]]) {
          $Parameter[[A]]->$Method_abstract_virtual[[pure]]();
          $Parameter[[A]]->$Method_virtual[[impl]]();
      }
      )cpp",
      R"cpp(
      <:[deprecated]:> int $Variable_def_deprecated[[x]];
      )cpp",
      R"cpp(
        // ObjC: Classes and methods
        @class $Class_decl[[Forward]];

        @interface $Class_def[[Foo]]
        @end
        @interface $Class_def[[Bar]] : $Class[[Foo]]
        -(id) $Method_decl[[x]]:(int)$Parameter_def[[a]] $Method_decl[[y]]:(int)$Parameter_def[[b]];
        +(instancetype)$StaticMethod_decl_static[[sharedInstance]];
        +(void) $StaticMethod_decl_static[[explode]];
        @end
        @implementation $Class_def[[Bar]]
        -(id) $Method_def[[x]]:(int)$Parameter_def[[a]] $Method_def[[y]]:(int)$Parameter_def[[b]] {
          return self;
        }
        +(instancetype)$StaticMethod_def_static[[sharedInstance]] { return 0; }
        +(void) $StaticMethod_def_static[[explode]] {}
        @end

        void $Function_def[[m]]($Class[[Bar]] *$Parameter_def[[b]]) {
          [$Parameter[[b]] $Method[[x]]:1 $Method[[y]]:2];
          [$Class[[Bar]] $StaticMethod_static[[explode]]];
        }
      )cpp",
      R"cpp(
        // ObjC: Protocols
        @protocol $Interface_def[[Protocol]]
        @end
        @protocol $Interface_def[[Protocol2]] <$Interface[[Protocol]]>
        @end
        @interface $Class_def[[Klass]] <$Interface[[Protocol]]>
        @end
        id<$Interface[[Protocol]]> $Variable_def[[x]];
      )cpp",
      R"cpp(
        // ObjC: Categories
        @interface $Class_def[[Foo]]
        @end
        @interface $Class[[Foo]]($Namespace_def[[Bar]])
        @end
        @implementation $Class[[Foo]]($Namespace_def[[Bar]])
        @end
      )cpp",
      R"cpp(
        // ObjC: Properties and Ivars.
        @interface $Class_def[[Foo]] {
          int $Field_decl[[_someProperty]];
        }
        @property(nonatomic, assign) int $Field_decl[[someProperty]];
        @property(readonly, class) $Class[[Foo]] *$Field_decl_readonly_static[[sharedInstance]];
        @end
        @implementation $Class_def[[Foo]]
        @synthesize someProperty = _someProperty;
        - (int)$Method_def[[otherMethod]] {
          return 0;
        }
        - (int)$Method_def[[doSomething]] {
          $Class[[Foo]].$Field_static[[sharedInstance]].$Field[[someProperty]] = 1;
          self.$Field[[someProperty]] = self.$Field[[someProperty]] + self.$Field[[otherMethod]] + 1;
          self->$Field[[_someProperty]] = $Field[[_someProperty]] + 1;
        }
        @end
      )cpp",
      // Member imported from dependent base
      R"cpp(
        template <typename> struct $Class_def[[Base]] {
          int $Field_decl[[member]];
        };
        template <typename $TemplateParameter_def[[T]]>
        struct $Class_def[[Derived]] : $Class[[Base]]<$TemplateParameter[[T]]> {
          using $Class[[Base]]<$TemplateParameter[[T]]>::$Field_dependentName[[member]];

          void $Method_def[[method]]() {
            (void)$Field_dependentName[[member]];
          }
        };
      )cpp",
      // Modifier for variables passed as non-const references
      R"cpp(
        struct $Class_def[[ClassWithOp]] {
            void operator()(int);
            void operator()(int, int &);
            void operator()(int, int, const int &);
            int &operator[](int &);
            int operator[](int) const;
        };
        struct $Class_def[[ClassWithStaticMember]] {
            static inline int $StaticField_def_static[[j]] = 0;
        };
        struct $Class_def[[ClassWithRefMembers]] {
          $Class_def_constrDestr[[ClassWithRefMembers]](int $Parameter_def[[i]])
            : $Field[[i1]]($Parameter[[i]]),
              $Field_readonly[[i2]]($Parameter[[i]]),
              $Field[[i3]]($Parameter_usedAsMutableReference[[i]]),
              $Field_readonly[[i4]]($Class[[ClassWithStaticMember]]::$StaticField_static[[j]]),
              $Field[[i5]]($Class[[ClassWithStaticMember]]::$StaticField_static_usedAsMutableReference[[j]])
          {}
          int $Field_decl[[i1]];
          const int &$Field_decl_readonly[[i2]];
          int &$Field_decl[[i3]];
          const int &$Field_decl_readonly[[i4]];
          int &$Field_decl[[i5]];
        };
        void $Function_def[[fun]](int, const int,
                                   int*, const int*,
                                   int&, const int&,
                                   int*&, const int*&, const int* const &,
                                   int**, int**&, int** const &,
                                   int = 123) {
          int $LocalVariable_def[[val]];
          int* $LocalVariable_def[[ptr]];
          const int* $LocalVariable_def_readonly[[constPtr]];
          int** $LocalVariable_def[[array]];
          $Function[[fun]]($LocalVariable[[val]], $LocalVariable[[val]],
                           $LocalVariable_usedAsMutablePointer[[ptr]], $LocalVariable_readonly[[constPtr]],
                           $LocalVariable_usedAsMutableReference[[val]], $LocalVariable[[val]],

                           $LocalVariable_usedAsMutableReference[[ptr]],
                           $LocalVariable_readonly_usedAsMutableReference[[constPtr]],
                           $LocalVariable_readonly[[constPtr]],

                           $LocalVariable_usedAsMutablePointer[[array]], $LocalVariable_usedAsMutableReference[[array]],
                           $LocalVariable[[array]]
                           );
          [](int){}($LocalVariable[[val]]);
          [](int&){}($LocalVariable_usedAsMutableReference[[val]]);
          [](const int&){}($LocalVariable[[val]]);
          $Class[[ClassWithOp]] $LocalVariable_def[[c]];
          const $Class[[ClassWithOp]] $LocalVariable_def_readonly[[c2]];
          $LocalVariable[[c]]($LocalVariable[[val]]);
          $LocalVariable[[c]](0, $LocalVariable_usedAsMutableReference[[val]]);
          $LocalVariable[[c]](0, 0, $LocalVariable[[val]]);
          $LocalVariable[[c]][$LocalVariable_usedAsMutableReference[[val]]];
          $LocalVariable_readonly[[c2]][$LocalVariable[[val]]];
        }
        struct $Class_def[[S]] {
          $Class_def_constrDestr[[S]](int&) {
            $Class[[S]] $LocalVariable_def[[s1]]($Field_usedAsMutableReference[[field]]);
            $Class[[S]] $LocalVariable_def[[s2]]($LocalVariable[[s1]].$Field_usedAsMutableReference[[field]]);

            $Class[[S]] $LocalVariable_def[[s3]]($StaticField_static_usedAsMutableReference[[staticField]]);
            $Class[[S]] $LocalVariable_def[[s4]]($Class[[S]]::$StaticField_static_usedAsMutableReference[[staticField]]);
          }
          int $Field_decl[[field]];
          static int $StaticField_decl_static[[staticField]];
        };
        template <typename $TemplateParameter_def[[X]]>
        void $Function_def[[foo]]($TemplateParameter[[X]]& $Parameter_def[[x]]) {
          // We do not support dependent types, so this one should *not* get the modifier.
          $Function[[foo]]($Parameter[[x]]);
        }
      )cpp",
      // init-captures
      R"cpp(
        void $Function_def[[foo]]() {
          int $LocalVariable_def[[a]], $LocalVariable_def[[b]];
          [ $LocalVariable_def[[c]] = $LocalVariable[[a]],
            $LocalVariable_def[[d]]($LocalVariable[[b]]) ]() {}();
        }
      )cpp",
      // Enum base specifier
      R"cpp(
        using $Primitive_decl[[MyTypedef]] = int;
        enum $Enum_decl[[MyEnum]] : $Primitive[[MyTypedef]] {};
      )cpp",
      // Enum base specifier
      R"cpp(
        typedef int $Primitive_decl[[MyTypedef]];
        enum $Enum_decl[[MyEnum]] : $Primitive[[MyTypedef]] {};
      )cpp",
      // Using enum
      R"cpp(
      enum class $Enum_decl[[Color]] { $EnumConstant_decl_readonly[[Black]] };
      namespace $Namespace_decl[[ns]] {
        using enum $Enum[[Color]];
        $Enum[[Color]] $Variable_def[[ModelT]] = $EnumConstant[[Black]];
      }
      )cpp",
      // Issue 1096
      R"cpp(
        void $Function_decl[[Foo]]();
        // Use <: :> digraphs for deprecated attribute to avoid conflict with annotation syntax
        <:<:deprecated:>:> void $Function_decl_deprecated[[Foo]](int* $Parameter_def[[x]]);
        void $Function_decl[[Foo]](int $Parameter_def[[x]]);
        template <typename $TemplateParameter_def[[T]]>
        void $Function_def[[Bar]]($TemplateParameter[[T]] $Parameter_def[[x]]) {
            $Function_deprecated[[Foo]]($Parameter[[x]]);
            $Function_deprecated[[Foo]]($Parameter[[x]]);
            $Function_deprecated[[Foo]]($Parameter[[x]]);
        }
      )cpp",
      // Predefined identifiers
      R"cpp(
        void $Function_def[[Foo]]() {
            const char *$LocalVariable_def_readonly[[s]] = $LocalVariable_readonly_static[[__func__]];
        }
      )cpp",
      // override and final
      R"cpp(
        class $Class_def_abstract[[Base]] { virtual void $Method_decl_abstract_virtual[[m]]() = 0; };
        class $Class_def[[override]] : public $Class_abstract[[Base]] { void $Method_decl_virtual[[m]]() $Modifier[[override]]; };
        class $Class_def[[final]] : public $Class[[override]] { void $Method_decl_virtual[[m]]() $Modifier[[override]] $Modifier[[final]]; };
      )cpp",
      // Issue 1222: readonly modifier for generic parameter
      R"cpp(
        template <typename $TemplateParameter_def[[T]]>
        auto $Function_def[[foo]](const $TemplateParameter[[T]] $Parameter_def_readonly[[template_type]], 
                                  const $TemplateParameter[[auto]] $Parameter_def_readonly[[auto_type]], 
                                  const int $Parameter_def_readonly[[explicit_type]]) {
            return $Parameter_readonly[[template_type]] 
                 + $Parameter_readonly[[auto_type]] 
                 + $Parameter_readonly[[explicit_type]];
        }
      )cpp",
      // Explicit template specialization
      R"cpp(
        struct $Class_def[[Base]]{};
        template <typename $TemplateParameter_def[[T]]>
        struct $Class_def[[S]] : public $Class[[Base]] {};
        template <>
        struct $Class_def[[S]]<void> : public $Class[[Base]] {};

        template <typename $TemplateParameter_def[[T]]>
        $TemplateParameter[[T]] $Variable_def[[x]] = {};
        template <>
        int $Variable_def[[x]]<int> = (int)sizeof($Class[[Base]]);
      )cpp",
      // no crash
      R"cpp(
        struct $Class_def[[Foo]] {
          void $Method_decl[[foo]]();
        };

        void $Function_def[[s]]($Class[[Foo]] $Parameter_def[[f]]) {
          auto $LocalVariable_def[[k]] = &$Class[[Foo]]::$Method[[foo]];
          ($Parameter[[f]].*$LocalVariable[[k]])(); // no crash on VisitCXXMemberCallExpr
        }
      )cpp"};
  for (const auto &TestCase : TestCases)
    // Mask off scope modifiers to keep the tests manageable.
    // They're tested separately.
    checkHighlightings(TestCase, {}, ~ScopeModifierMask);

  checkHighlightings(R"cpp(
    class $Class_def[[A]] {
      #include "imp.h"
    };
  )cpp",
                     {{"imp.h", R"cpp(
    int someMethod();
    void otherMethod();
  )cpp"}},
                     ~ScopeModifierMask);

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
  )cpp"}},
                     ~ScopeModifierMask);

  checkHighlightings(R"cpp(
    #include "SYSObject.h"
    @interface $Class_defaultLibrary[[SYSObject]] ($Namespace_def[[UserCategory]])
    @property(nonatomic, readonly) int $Field_decl_readonly[[user_property]];
    @end
    int $Function_def[[somethingUsingSystemSymbols]]() {
      $Class_defaultLibrary[[SYSObject]] *$LocalVariable_def[[obj]] = [$Class_defaultLibrary[[SYSObject]] $StaticMethod_static_defaultLibrary[[new]]];
      return $LocalVariable[[obj]].$Field_defaultLibrary[[value]] + $LocalVariable[[obj]].$Field_readonly[[user_property]];
    }
  )cpp",
                     {{"SystemSDK/SYSObject.h", R"cpp(
    @interface SYSObject
    @property(nonatomic, assign) int value;
    + (instancetype)new;
    @end
  )cpp"}},
                     ~ScopeModifierMask, {"-isystemSystemSDK/"});
}

TEST(SemanticHighlighting, ScopeModifiers) {
  const char *TestCases[] = {
      R"cpp(
        static int $Variable_fileScope[[x]];
        namespace $Namespace_globalScope[[ns]] {
          class $Class_globalScope[[x]];
        }
        namespace {
          void $Function_fileScope[[foo]]();
        }
      )cpp",
      R"cpp(
        void $Function_globalScope[[foo]](int $Parameter_functionScope[[y]]) {
          int $LocalVariable_functionScope[[z]];
        }
      )cpp",
      R"cpp(
        // Lambdas are considered functions, not classes.
        auto $Variable_fileScope[[x]] = [$LocalVariable_functionScope[[m]](42)] {
          return $LocalVariable_functionScope[[m]];
        };
      )cpp",
      R"cpp(
        // Classes in functions are classes.
        void $Function_globalScope[[foo]]() {
          class $Class_functionScope[[X]] {
            int $Field_classScope[[x]];
          };
        };
      )cpp",
      R"cpp(
        template <int $TemplateParameter_classScope[[T]]>
        class $Class_globalScope[[X]] {
        };
      )cpp",
      R"cpp(
        // No useful scope for template parameters of variable templates.
        template <typename $TemplateParameter[[A]]>
        unsigned $Variable_globalScope[[X]] =
          $TemplateParameter[[A]]::$Unknown_classScope[[x]];
      )cpp",
      R"cpp(
        #define $Macro_globalScope[[X]] 1
        int $Variable_globalScope[[Y]] = $Macro_globalScope[[X]];
      )cpp",
  };

  for (const char *Test : TestCases)
    checkHighlightings(Test, {}, ScopeModifierMask);
}

// Ranges are highlighted as variables, unless highlighted as $Function etc.
std::vector<HighlightingToken> tokens(llvm::StringRef MarkedText) {
  Annotations A(MarkedText);
  std::vector<HighlightingToken> Results;
  for (const Range& R : A.ranges())
    Results.push_back({HighlightingKind::Variable, 0, R});
  for (unsigned I = 0; I < static_cast<unsigned>(HighlightingKind::LastKind); ++I) {
    HighlightingKind Kind = static_cast<HighlightingKind>(I);
    for (const Range& R : A.ranges(llvm::to_string(Kind)))
      Results.push_back({Kind, 0, R});
  }
  llvm::sort(Results);
  return Results;
}

TEST(SemanticHighlighting, toSemanticTokens) {
  auto Tokens = tokens(R"(
 [[blah]]

    $Function[[big]] [[bang]]
  )");
  Tokens.front().Modifiers |= unsigned(HighlightingModifier::Declaration);
  Tokens.front().Modifiers |= unsigned(HighlightingModifier::Readonly);
  auto Results = toSemanticTokens(Tokens, /*Code=*/"");

  ASSERT_THAT(Results, SizeIs(3));
  EXPECT_EQ(Results[0].tokenType, unsigned(HighlightingKind::Variable));
  EXPECT_EQ(Results[0].tokenModifiers,
            unsigned(HighlightingModifier::Declaration) |
                unsigned(HighlightingModifier::Readonly));
  EXPECT_EQ(Results[0].deltaLine, 1u);
  EXPECT_EQ(Results[0].deltaStart, 1u);
  EXPECT_EQ(Results[0].length, 4u);

  EXPECT_EQ(Results[1].tokenType, unsigned(HighlightingKind::Function));
  EXPECT_EQ(Results[1].tokenModifiers, 0u);
  EXPECT_EQ(Results[1].deltaLine, 2u);
  EXPECT_EQ(Results[1].deltaStart, 4u);
  EXPECT_EQ(Results[1].length, 3u);

  EXPECT_EQ(Results[2].tokenType, unsigned(HighlightingKind::Variable));
  EXPECT_EQ(Results[1].tokenModifiers, 0u);
  EXPECT_EQ(Results[2].deltaLine, 0u);
  EXPECT_EQ(Results[2].deltaStart, 4u);
  EXPECT_EQ(Results[2].length, 4u);
}

TEST(SemanticHighlighting, diffSemanticTokens) {
  auto Before = toSemanticTokens(tokens(R"(
    [[foo]] [[bar]] [[baz]]
    [[one]] [[two]] [[three]]
  )"),
                                 /*Code=*/"");
  EXPECT_THAT(diffTokens(Before, Before), IsEmpty());

  auto After = toSemanticTokens(tokens(R"(
    [[foo]] [[hello]] [[world]] [[baz]]
    [[one]] [[two]] [[three]]
  )"),
                                /*Code=*/"");

  // Replace [bar, baz] with [hello, world, baz]
  auto Diff = diffTokens(Before, After);
  ASSERT_THAT(Diff, SizeIs(1));
  EXPECT_EQ(1u, Diff.front().startToken);
  EXPECT_EQ(2u, Diff.front().deleteTokens);
  ASSERT_THAT(Diff.front().tokens, SizeIs(3));
  // hello
  EXPECT_EQ(0u, Diff.front().tokens[0].deltaLine);
  EXPECT_EQ(4u, Diff.front().tokens[0].deltaStart);
  EXPECT_EQ(5u, Diff.front().tokens[0].length);
  // world
  EXPECT_EQ(0u, Diff.front().tokens[1].deltaLine);
  EXPECT_EQ(6u, Diff.front().tokens[1].deltaStart);
  EXPECT_EQ(5u, Diff.front().tokens[1].length);
  // baz
  EXPECT_EQ(0u, Diff.front().tokens[2].deltaLine);
  EXPECT_EQ(6u, Diff.front().tokens[2].deltaStart);
  EXPECT_EQ(3u, Diff.front().tokens[2].length);
}

TEST(SemanticHighlighting, MultilineTokens) {
  llvm::StringRef AnnotatedCode = R"cpp(
  [[fo
o
o]] [[bar]])cpp";
  auto Toks = toSemanticTokens(tokens(AnnotatedCode),
                               Annotations(AnnotatedCode).code());
  ASSERT_THAT(Toks, SizeIs(4));
  // foo
  EXPECT_EQ(Toks[0].deltaLine, 1u);
  EXPECT_EQ(Toks[0].deltaStart, 2u);
  EXPECT_EQ(Toks[0].length, 2u);
  EXPECT_EQ(Toks[1].deltaLine, 1u);
  EXPECT_EQ(Toks[1].deltaStart, 0u);
  EXPECT_EQ(Toks[1].length, 1u);
  EXPECT_EQ(Toks[2].deltaLine, 1u);
  EXPECT_EQ(Toks[2].deltaStart, 0u);
  EXPECT_EQ(Toks[2].length, 1u);

  // bar
  EXPECT_EQ(Toks[3].deltaLine, 0u);
  EXPECT_EQ(Toks[3].deltaStart, 2u);
  EXPECT_EQ(Toks[3].length, 3u);
}
} // namespace
} // namespace clangd
} // namespace clang
