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
  assert(std::is_sorted(
      Tokens.begin(), Tokens.end(),
      [](const HighlightingToken &L, const HighlightingToken &R) {
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

    OS << Input.substr(NextChar, StartOffset - NextChar);
    OS << '$' << T.Kind;
    for (unsigned I = 0;
         I <= static_cast<uint32_t>(HighlightingModifier::LastModifier); ++I) {
      if (T.Modifiers & (1 << I))
        OS << '_' << static_cast<HighlightingModifier>(I);
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
      struct $Class_decl[[AS]] {
        double $Field_decl[[SomeMember]];
      };
      struct {
      } $Variable_decl[[S]];
      void $Function_decl[[foo]](int $Parameter_decl[[A]], $Class[[AS]] $Parameter_decl[[As]]) {
        $Primitive_deduced_defaultLibrary[[auto]] $LocalVariable_decl[[VeryLongVariableName]] = 12312;
        $Class[[AS]]     $LocalVariable_decl[[AA]];
        $Primitive_deduced_defaultLibrary[[auto]] $LocalVariable_decl[[L]] = $LocalVariable[[AA]].$Field[[SomeMember]] + $Parameter[[A]];
        auto $LocalVariable_decl[[FN]] = [ $LocalVariable[[AA]]](int $Parameter_decl[[A]]) -> void {};
        $LocalVariable[[FN]](12312);
      }
    )cpp",
      R"cpp(
      void $Function_decl[[foo]](int);
      void $Function_decl[[Gah]]();
      void $Function_decl[[foo]]() {
        auto $LocalVariable_decl[[Bou]] = $Function[[Gah]];
      }
      struct $Class_decl[[A]] {
        void $Method_decl[[abc]]();
      };
    )cpp",
      R"cpp(
      namespace $Namespace_decl[[abc]] {
        template<typename $TemplateParameter_decl[[T]]>
        struct $Class_decl[[A]] {
          $TemplateParameter[[T]] $Field_decl[[t]];
        };
      }
      template<typename $TemplateParameter_decl[[T]]>
      struct $Class_decl[[C]] : $Namespace[[abc]]::$Class[[A]]<$TemplateParameter[[T]]> {
        typename $TemplateParameter[[T]]::$Type_dependentName[[A]]* $Field_decl[[D]];
      };
      $Namespace[[abc]]::$Class[[A]]<int> $Variable_decl[[AA]];
      typedef $Namespace[[abc]]::$Class[[A]]<int> $Class_decl[[AAA]];
      struct $Class_decl[[B]] {
        $Class_decl[[B]]();
        ~$Class[[B]](); // FIXME: inconsistent with constructor
        void operator<<($Class[[B]]);
        $Class[[AAA]] $Field_decl[[AA]];
      };
      $Class[[B]]::$Class_decl[[B]]() {}
      $Class[[B]]::~$Class[[B]]() {} // FIXME: inconsistent with constructor
      void $Function_decl[[f]] () {
        $Class[[B]] $LocalVariable_decl[[BB]] = $Class[[B]]();
        $LocalVariable[[BB]].~$Class[[B]]();
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
      struct $Class_decl[[A]] {
        $Enum[[E]] $Field_decl[[EEE]];
        $Enum[[EE]] $Field_decl[[EEEE]];
      };
      int $Variable_decl[[I]] = $EnumConstant_readonly[[Hi]];
      $Enum[[E]] $Variable_decl[[L]] = $Enum[[E]]::$EnumConstant_readonly[[B]];
    )cpp",
      R"cpp(
      namespace $Namespace_decl[[abc]] {
        namespace {}
        namespace $Namespace_decl[[bcd]] {
          struct $Class_decl[[A]] {};
          namespace $Namespace_decl[[cde]] {
            struct $Class_decl[[A]] {
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
      $Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable_decl[[AA]];
      $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]] $Variable_decl[[AAA]] =
            $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]]::$EnumConstant_readonly[[Hi]];
      ::$Namespace[[vwz]]::$Class[[A]] $Variable_decl[[B]];
      ::$Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable_decl[[BB]];
    )cpp",
      R"cpp(
      struct $Class_decl[[D]] {
        double $Field_decl[[C]];
      };
      struct $Class_decl[[A]] {
        double $Field_decl[[B]];
        $Class[[D]] $Field_decl[[E]];
        static double $StaticField_decl_static[[S]];
        static void $StaticMethod_decl_static[[bar]]() {}
        void $Method_decl[[foo]]() {
          $Field[[B]] = 123;
          this->$Field[[B]] = 156;
          this->$Method[[foo]]();
          $Method[[foo]]();
          $StaticMethod_static[[bar]]();
          $StaticField_static[[S]] = 90.1;
        }
      };
      void $Function_decl[[foo]]() {
        $Class[[A]] $LocalVariable_decl[[AA]];
        $LocalVariable[[AA]].$Field[[B]] += 2;
        $LocalVariable[[AA]].$Method[[foo]]();
        $LocalVariable[[AA]].$Field[[E]].$Field[[C]];
        $Class[[A]]::$StaticField_static[[S]] = 90;
      }
    )cpp",
      R"cpp(
      struct $Class_decl[[AA]] {
        int $Field_decl[[A]];
      };
      int $Variable_decl[[B]];
      $Class[[AA]] $Variable_decl[[A]]{$Variable[[B]]};
    )cpp",
      R"cpp(
      namespace $Namespace_decl[[a]] {
        struct $Class_decl[[A]] {};
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
      template<typename $TemplateParameter_decl[[T]], typename = void>
      class $Class_decl[[A]] {
        $TemplateParameter[[T]] $Field_decl[[AA]];
        $TemplateParameter[[T]] $Method_decl[[foo]]();
      };
      template<class $TemplateParameter_decl[[TT]]>
      class $Class_decl[[B]] {
        $Class[[A]]<$TemplateParameter[[TT]]> $Field_decl[[AA]];
      };
      template<class $TemplateParameter_decl[[TT]], class $TemplateParameter_decl[[GG]]>
      class $Class_decl[[BB]] {};
      template<class $TemplateParameter_decl[[T]]>
      class $Class_decl[[BB]]<$TemplateParameter[[T]], int> {};
      template<class $TemplateParameter_decl[[T]]>
      class $Class_decl[[BB]]<$TemplateParameter[[T]], $TemplateParameter[[T]]*> {};

      template<template<class> class $TemplateParameter_decl[[T]], class $TemplateParameter_decl[[C]]>
      $TemplateParameter[[T]]<$TemplateParameter[[C]]> $Function_decl[[f]]();

      template<typename>
      class $Class_decl[[Foo]] {};

      template<typename $TemplateParameter_decl[[T]]>
      void $Function_decl[[foo]]($TemplateParameter[[T]] ...);
    )cpp",
      R"cpp(
      template <class $TemplateParameter_decl[[T]]>
      struct $Class_decl[[Tmpl]] {$TemplateParameter[[T]] $Field_decl[[x]] = 0;};
      extern template struct $Class_decl[[Tmpl]]<float>;
      template struct $Class_decl[[Tmpl]]<double>;
    )cpp",
      // This test is to guard against highlightings disappearing when using
      // conversion operators as their behaviour in the clang AST differ from
      // other CXXMethodDecls.
      R"cpp(
      class $Class_decl[[Foo]] {};
      struct $Class_decl[[Bar]] {
        explicit operator $Class[[Foo]]*() const;
        explicit operator int() const;
        operator $Class[[Foo]]();
      };
      void $Function_decl[[f]]() {
        $Class[[Bar]] $LocalVariable_decl[[B]];
        $Class[[Foo]] $LocalVariable_decl[[F]] = $LocalVariable[[B]];
        $Class[[Foo]] *$LocalVariable_decl[[FP]] = ($Class[[Foo]]*)$LocalVariable[[B]];
        int $LocalVariable_decl[[I]] = (int)$LocalVariable[[B]];
      }
    )cpp",
      R"cpp(
      struct $Class_decl[[B]] {};
      struct $Class_decl[[A]] {
        $Class[[B]] $Field_decl[[BB]];
        $Class[[A]] &operator=($Class[[A]] &&$Parameter_decl[[O]]);
      };

      $Class[[A]] &$Class[[A]]::operator=($Class[[A]] &&$Parameter_decl[[O]]) = default;
    )cpp",
      R"cpp(
      enum $Enum_decl[[En]] {
        $EnumConstant_decl_readonly[[EC]],
      };
      class $Class_decl[[Foo]] {};
      class $Class_decl[[Bar]] {
      public:
        $Class[[Foo]] $Field_decl[[Fo]];
        $Enum[[En]] $Field_decl[[E]];
        int $Field_decl[[I]];
        $Class_decl[[Bar]] ($Class[[Foo]] $Parameter_decl[[F]],
                $Enum[[En]] $Parameter_decl[[E]])
        : $Field[[Fo]] ($Parameter[[F]]), $Field[[E]] ($Parameter[[E]]),
          $Field[[I]] (123) {}
      };
      class $Class_decl[[Bar2]] : public $Class[[Bar]] {
        $Class_decl[[Bar2]]() : $Class[[Bar]]($Class[[Foo]](), $EnumConstant_readonly[[EC]]) {}
      };
    )cpp",
      R"cpp(
      enum $Enum_decl[[E]] {
        $EnumConstant_decl_readonly[[E]],
      };
      class $Class_decl[[Foo]] {};
      $Enum_deduced[[auto]] $Variable_decl[[AE]] = $Enum[[E]]::$EnumConstant_readonly[[E]];
      $Class_deduced[[auto]] $Variable_decl[[AF]] = $Class[[Foo]]();
      $Class_deduced[[decltype]](auto) $Variable_decl[[AF2]] = $Class[[Foo]]();
      $Class_deduced[[auto]] *$Variable_decl[[AFP]] = &$Variable[[AF]];
      $Enum_deduced[[auto]] &$Variable_decl[[AER]] = $Variable[[AE]];
      $Primitive_deduced_defaultLibrary[[auto]] $Variable_decl[[Form]] = 10.2 + 2 * 4;
      $Primitive_deduced_defaultLibrary[[decltype]]($Variable[[Form]]) $Variable_decl[[F]] = 10;
      auto $Variable_decl[[Fun]] = []()->void{};
    )cpp",
      R"cpp(
      class $Class_decl[[G]] {};
      template<$Class[[G]] *$TemplateParameter_decl_readonly[[U]]>
      class $Class_decl[[GP]] {};
      template<$Class[[G]] &$TemplateParameter_decl_readonly[[U]]>
      class $Class_decl[[GR]] {};
      template<int *$TemplateParameter_decl_readonly[[U]]>
      class $Class_decl[[IP]] {
        void $Method_decl[[f]]() {
          *$TemplateParameter_readonly[[U]] += 5;
        }
      };
      template<unsigned $TemplateParameter_decl_readonly[[U]] = 2>
      class $Class_decl[[Foo]] {
        void $Method_decl[[f]]() {
          for(int $LocalVariable_decl[[I]] = 0;
            $LocalVariable[[I]] < $TemplateParameter_readonly[[U]];) {}
        }
      };

      $Class[[G]] $Variable_decl[[L]];
      void $Function_decl[[f]]() {
        $Class[[Foo]]<123> $LocalVariable_decl[[F]];
        $Class[[GP]]<&$Variable[[L]]> $LocalVariable_decl[[LL]];
        $Class[[GR]]<$Variable[[L]]> $LocalVariable_decl[[LLL]];
      }
    )cpp",
      R"cpp(
      template<typename $TemplateParameter_decl[[T]],
        void ($TemplateParameter[[T]]::*$TemplateParameter_decl_readonly[[method]])(int)>
      struct $Class_decl[[G]] {
        void $Method_decl[[foo]](
            $TemplateParameter[[T]] *$Parameter_decl[[O]]) {
          ($Parameter[[O]]->*$TemplateParameter_readonly[[method]])(10);
        }
      };
      struct $Class_decl[[F]] {
        void $Method_decl[[f]](int);
      };
      template<void (*$TemplateParameter_decl_readonly[[Func]])()>
      struct $Class_decl[[A]] {
        void $Method_decl[[f]]() {
          (*$TemplateParameter_readonly[[Func]])();
        }
      };

      void $Function_decl[[foo]]() {
        $Class[[F]] $LocalVariable_decl[[FF]];
        $Class[[G]]<$Class[[F]], &$Class[[F]]::$Method[[f]]> $LocalVariable_decl[[GG]];
        $LocalVariable[[GG]].$Method[[foo]](&$LocalVariable[[FF]]);
        $Class[[A]]<$Function[[foo]]> $LocalVariable_decl[[AA]];
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
      $Macro[[DEF_CLASS]]($Class_decl[[A]])
      #define $Macro_decl[[MACRO_CONCAT]](X, V, T) T foo##X = V
      #define $Macro_decl[[DEF_VAR]](X, V) int X = V
      #define $Macro_decl[[DEF_VAR_T]](T, X, V) T X = V
      #define $Macro_decl[[DEF_VAR_REV]](V, X) DEF_VAR(X, V)
      #define $Macro_decl[[CPY]](X) X
      #define $Macro_decl[[DEF_VAR_TYPE]](X, Y) X Y
      #define $Macro_decl[[SOME_NAME]] variable
      #define $Macro_decl[[SOME_NAME_SET]] variable2 = 123
      #define $Macro_decl[[INC_VAR]](X) X += 2
      void $Function_decl[[foo]]() {
        $Macro[[DEF_VAR]]($LocalVariable_decl[[X]],  123);
        $Macro[[DEF_VAR_REV]](908, $LocalVariable_decl[[XY]]);
        int $Macro[[CPY]]( $LocalVariable_decl[[XX]] );
        $Macro[[DEF_VAR_TYPE]]($Class[[A]], $LocalVariable_decl[[AA]]);
        double $Macro[[SOME_NAME]];
        int $Macro[[SOME_NAME_SET]];
        $LocalVariable[[variable]] = 20.1;
        $Macro[[MACRO_CONCAT]](var, 2, float);
        $Macro[[DEF_VAR_T]]($Class[[A]], $Macro[[CPY]](
              $Macro[[CPY]]($LocalVariable_decl[[Nested]])),
            $Macro[[CPY]]($Class[[A]]()));
        $Macro[[INC_VAR]]($LocalVariable[[variable]]);
      }
      void $Macro[[SOME_NAME]]();
      $Macro[[DEF_VAR]]($Variable_decl[[MMMMM]], 567);
      $Macro[[DEF_VAR_REV]](756, $Variable_decl[[AB]]);

      #define $Macro_decl[[CALL_FN]](F) F();
      #define $Macro_decl[[DEF_FN]](F) void F ()
      $Macro[[DEF_FN]]($Function_decl[[g]]) {
        $Macro[[CALL_FN]]($Function[[foo]]);
      }
    )cpp",
      R"cpp(
      #define $Macro_decl[[fail]](expr) expr
      #define $Macro_decl[[assert]](COND) if (!(COND)) { fail("assertion failed" #COND); }
      // Preamble ends.
      int $Variable_decl[[x]];
      int $Variable_decl[[y]];
      int $Function_decl[[f]]();
      void $Function_decl[[foo]]() {
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
      struct $Class_decl[[S]] {
        float $Field_decl[[Value]];
        $Class[[S]] *$Field_decl[[Next]];
      };
      $Class[[S]] $Variable_decl[[Global]][2] = {$Class[[S]](), $Class[[S]]()};
      auto [$Variable_decl[[G1]], $Variable_decl[[G2]]] = $Variable[[Global]];
      void $Function_decl[[f]]($Class[[S]] $Parameter_decl[[P]]) {
        int $LocalVariable_decl[[A]][2] = {1,2};
        auto [$LocalVariable_decl[[B1]], $LocalVariable_decl[[B2]]] = $LocalVariable[[A]];
        auto [$LocalVariable_decl[[G1]], $LocalVariable_decl[[G2]]] = $Variable[[Global]];
        $Class_deduced[[auto]] [$LocalVariable_decl[[P1]], $LocalVariable_decl[[P2]]] = $Parameter[[P]];
        // Highlights references to BindingDecls.
        $LocalVariable[[B1]]++;
      }
    )cpp",
      R"cpp(
      template<class $TemplateParameter_decl[[T]]>
      class $Class_decl[[A]] {
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
      template <class $TemplateParameter_decl[[T]]>
      void $Function_decl[[phase1]]($TemplateParameter[[T]]);
      template <class $TemplateParameter_decl[[T]]>
      void $Function_decl[[foo]]($TemplateParameter[[T]] $Parameter_decl[[P]]) {
        $Function[[phase1]]($Parameter[[P]]);
        $Unknown_dependentName[[phase2]]($Parameter[[P]]);
      }
    )cpp",
      R"cpp(
      class $Class_decl[[A]] {
        template <class $TemplateParameter_decl[[T]]>
        void $Method_decl[[bar]]($TemplateParameter[[T]]);
      };

      template <class $TemplateParameter_decl[[U]]>
      void $Function_decl[[foo]]($TemplateParameter[[U]] $Parameter_decl[[P]]) {
        $Class[[A]]().$Method[[bar]]($Parameter[[P]]);
      }
    )cpp",
      R"cpp(
      struct $Class_decl[[A]] {
        template <class $TemplateParameter_decl[[T]]>
        static void $StaticMethod_decl_static[[foo]]($TemplateParameter[[T]]);
      };

      template <class $TemplateParameter_decl[[T]]>
      struct $Class_decl[[B]] {
        void $Method_decl[[bar]]() {
          $Class[[A]]::$StaticMethod_static[[foo]]($TemplateParameter[[T]]());
        }
      };
    )cpp",
      R"cpp(
      template <class $TemplateParameter_decl[[T]]>
      void $Function_decl[[foo]](typename $TemplateParameter[[T]]::$Type_dependentName[[Type]]
                                            = $TemplateParameter[[T]]::$Unknown_dependentName[[val]]);
    )cpp",
      R"cpp(
      template <class $TemplateParameter_decl[[T]]>
      void $Function_decl[[foo]]($TemplateParameter[[T]] $Parameter_decl[[P]]) {
        $Parameter[[P]].$Unknown_dependentName[[Field]];
      }
    )cpp",
      R"cpp(
      template <class $TemplateParameter_decl[[T]]>
      class $Class_decl[[A]] {
        int $Method_decl[[foo]]() {
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
      template <template <class> class $TemplateParameter_decl[[TT]],
                template <class> class ...$TemplateParameter_decl[[TTs]]>
      struct $Class_decl[[Foo]] {
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
      int $Variable_decl[[EndPreamble]];

      // Code after the preamble.
      // Code inside inactive blocks does not get regular highlightings
      // because it's not part of the AST.
      #define $Macro_decl[[test2]]
$InactiveCode[[#if defined(test)]]
$InactiveCode[[int Inactive2;]]
$InactiveCode[[#elif defined(test2)]]
      int $Variable_decl[[Active1]];
$InactiveCode[[#else]]
$InactiveCode[[int Inactive3;]]
$InactiveCode[[#endif]]

      #ifndef $Macro[[test]]
      int $Variable_decl[[Active2]];
      #endif

$InactiveCode[[#ifdef test]]
$InactiveCode[[int Inactive4;]]
$InactiveCode[[#else]]
      int $Variable_decl[[Active3]];
      #endif
    )cpp",
      // Argument to 'sizeof...'
      R"cpp(
      template <typename... $TemplateParameter_decl[[Elements]]>
      struct $Class_decl[[TupleSize]] {
        static const int $StaticField_decl_readonly_static[[size]] =
sizeof...($TemplateParameter[[Elements]]);
      };
    )cpp",
      // More dependent types
      R"cpp(
      template <typename $TemplateParameter_decl[[T]]>
      struct $Class_decl[[Waldo]] {
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
      struct $Class_decl[[Foo]] {
        int $Field_decl[[Waldo]];
        void $Method_decl[[bar]]() {
          $Class[[Foo]]().$Field_dependentName[[Waldo]];
        }
        template <typename $TemplateParameter_decl[[U]]>
        void $Method_decl[[bar1]]() {
          $Class[[Foo]]<$TemplateParameter[[U]]>().$Field_dependentName[[Waldo]];
        }
      };
    )cpp",
      // Concepts
      R"cpp(
      template <typename $TemplateParameter_decl[[T]]>
      concept $Concept_decl[[Fooable]] = 
          requires($TemplateParameter[[T]] $Parameter_decl[[F]]) {
            $Parameter[[F]].$Unknown_dependentName[[foo]]();
          };
      template <typename $TemplateParameter_decl[[T]]>
          requires $Concept[[Fooable]]<$TemplateParameter[[T]]>
      void $Function_decl[[bar]]($TemplateParameter[[T]] $Parameter_decl[[F]]) {
        $Parameter[[F]].$Unknown_dependentName[[foo]]();
      }
    )cpp",
      // Dependent template name
      R"cpp(
      template <template <typename> class> struct $Class_decl[[A]] {};
      template <typename $TemplateParameter_decl[[T]]>
      using $Typedef_decl[[W]] = $Class[[A]]<
        $TemplateParameter[[T]]::template $Class_dependentName[[Waldo]]
      >;
    )cpp",
      R"cpp(
      class $Class_decl_abstract[[Abstract]] {
      public:
        virtual void $Method_decl_abstract_virtual[[pure]]() = 0;
        virtual void $Method_decl_virtual[[impl]]();
      };
      void $Function_decl[[foo]]($Class_abstract[[Abstract]]* $Parameter_decl[[A]]) {
          $Parameter[[A]]->$Method_abstract_virtual[[pure]]();
          $Parameter[[A]]->$Method_virtual[[impl]]();
      }
      )cpp",
      R"cpp(
      <:[deprecated]:> int $Variable_decl_deprecated[[x]];
      )cpp",
      R"cpp(
        // ObjC: Classes and methods
        @class $Class_decl[[Forward]];

        @interface $Class_decl[[Foo]]
        @end
        @interface $Class_decl[[Bar]] : $Class[[Foo]]
        -($Class[[id]]) $Method_decl[[x]]:(int)$Parameter_decl[[a]] $Method_decl[[y]]:(int)$Parameter_decl[[b]];
        +(void) $StaticMethod_decl_static[[explode]];
        @end
        @implementation $Class_decl[[Bar]]
        -($Class[[id]]) $Method_decl[[x]]:(int)$Parameter_decl[[a]] $Method_decl[[y]]:(int)$Parameter_decl[[b]] {
          return self;
        }
        +(void) $StaticMethod_decl_static[[explode]] {}
        @end

        void $Function_decl[[m]]($Class[[Bar]] *$Parameter_decl[[b]]) {
          [$Parameter[[b]] $Method[[x]]:1 $Method[[y]]:2];
          [$Class[[Bar]] $StaticMethod_static[[explode]]];
        }
      )cpp",
      R"cpp(
        // ObjC: Protocols
        @protocol $Interface_decl[[Protocol]]
        @end
        @protocol $Interface_decl[[Protocol2]] <$Interface[[Protocol]]>
        @end
        @interface $Class_decl[[Klass]] <$Interface[[Protocol]]>
        @end
        id<$Interface[[Protocol]]> $Variable_decl[[x]];
      )cpp",
      R"cpp(
        // ObjC: Categories
        @interface $Class_decl[[Foo]]
        @end
        @interface $Class[[Foo]]($Namespace_decl[[Bar]])
        @end
        @implementation $Class[[Foo]]($Namespace_decl[[Bar]])
        @end
      )cpp",
      R"cpp(
        // ObjC: Properties and Ivars.
        @interface $Class_decl[[Foo]] {
          int $Field_decl[[_someProperty]];
        }
        @property(nonatomic, assign) int $Field_decl[[someProperty]];
        @property(readonly, class) $Class[[Foo]] *$Field_decl_readonly_static[[sharedInstance]];
        @end
        @implementation $Class_decl[[Foo]]
        @synthesize someProperty = _someProperty;
        - (int)$Method_decl[[otherMethod]] {
          return 0;
        }
        - (int)$Method_decl[[doSomething]] {
          $Class[[Foo]].$Field_static[[sharedInstance]].$Field[[someProperty]] = 1;
          self.$Field[[someProperty]] = self.$Field[[someProperty]] + self.$Field[[otherMethod]] + 1;
          self->$Field[[_someProperty]] = $Field[[_someProperty]] + 1;
        }
        @end
      )cpp",
      // Member imported from dependent base
      R"cpp(
        template <typename> struct $Class_decl[[Base]] {
          int $Field_decl[[member]];
        };
        template <typename $TemplateParameter_decl[[T]]>
        struct $Class_decl[[Derived]] : $Class[[Base]]<$TemplateParameter[[T]]> {
          using $Class[[Base]]<$TemplateParameter[[T]]>::$Field_dependentName[[member]];

          void $Method_decl[[method]]() {
            (void)$Field_dependentName[[member]];
          }
        };
      )cpp",
  };
  for (const auto &TestCase : TestCases)
    // Mask off scope modifiers to keep the tests manageable.
    // They're tested separately.
    checkHighlightings(TestCase, {}, ~ScopeModifierMask);

  checkHighlightings(R"cpp(
    class $Class_decl[[A]] {
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
    @interface $Class_defaultLibrary[[SYSObject]] ($Namespace_decl[[UserCategory]])
    @property(nonatomic, readonly) int $Field_decl_readonly[[user_property]];
    @end
    int $Function_decl[[somethingUsingSystemSymbols]]() {
      $Class_defaultLibrary[[SYSObject]] *$LocalVariable_decl[[obj]] = [$Class_defaultLibrary[[SYSObject]] $StaticMethod_static_defaultLibrary[[new]]];
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
        auto $Variable_fileScope[[x]] = [m(42)] { // FIXME: annotate capture
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
  auto Results = toSemanticTokens(Tokens);

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
  )"));
  EXPECT_THAT(diffTokens(Before, Before), IsEmpty());

  auto After = toSemanticTokens(tokens(R"(
    [[foo]] [[hello]] [[world]] [[baz]]
    [[one]] [[two]] [[three]]
  )"));

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

} // namespace
} // namespace clangd
} // namespace clang
