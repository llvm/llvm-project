// RUN: %clang_cc1 -Wdocumentation -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

/// Aaa
int TestLocation;
// CHECK: VarDecl{{.*}}TestLocation
// CHECK-NEXT:   FullComment 0x{{[^ ]*}} <line:[[@LINE-3]]:4, col:7>

///
int TestIndent;
// CHECK:      {{^VarDecl.*TestIndent[^()]*$}}
// CHECK-NEXT: {{^`-FullComment.*>$}}

/// Aaa
int Test_TextComment;
// CHECK:      VarDecl{{.*}}Test_TextComment
// CHECK-NEXT:   FullComment
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" Aaa"

/// \brief Aaa
int Test_BlockCommandComment;
// CHECK:      VarDecl{{.*}}Test_BlockCommandComment
// CHECK:        BlockCommandComment{{.*}} Name="brief"
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" Aaa"

/// \param Aaa xxx
/// \param [in,out] Bbb yyy
void Test_ParamCommandComment(int Aaa, int Bbb);
// CHECK:      FunctionDecl{{.*}}Test_ParamCommandComment
// CHECK:        ParamCommandComment{{.*}} [in] implicitly Param="Aaa" ParamIndex=0
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" xxx"
// CHECK:        ParamCommandComment{{.*}} [in,out] explicitly Param="Bbb" ParamIndex=1
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" yyy"

/// \tparam Aaa xxx
template <typename Aaa> class Test_TParamCommandComment;
// CHECK:      ClassTemplateDecl{{.*}}Test_TParamCommandComment
// CHECK:        TParamCommandComment{{.*}} Param="Aaa" Position=<0>
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" xxx"

/// \c Aaa
int Test_InlineCommandComment;
// CHECK:      VarDecl{{.*}}Test_InlineCommandComment
// CHECK:        InlineCommandComment{{.*}} Name="c" RenderMonospaced Arg[0]="Aaa"

/// <a>Aaa</a>
/// <br/>
int Test_HTMLTagComment;
// CHECK:      VarDecl{{.*}}Test_HTMLTagComment
// CHECK-NEXT:   FullComment
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" "
// CHECK-NEXT:       HTMLStartTagComment{{.*}} Name="a"
// CHECK-NEXT:       TextComment{{.*}} Text="Aaa"
// CHECK-NEXT:       HTMLEndTagComment{{.*}} Name="a"
// CHECK-NEXT:       TextComment{{.*}} Text=" "
// CHECK-NEXT:       HTMLStartTagComment{{.*}} Name="br" SelfClosing

/// \verbatim
/// Aaa
/// \endverbatim
int Test_VerbatimBlockComment;
// CHECK:      VarDecl{{.*}}Test_VerbatimBlockComment
// CHECK:        VerbatimBlockComment{{.*}} Name="verbatim" CloseName="endverbatim"
// CHECK-NEXT:     VerbatimBlockLineComment{{.*}} Text=" Aaa"

/// \param ... More arguments
template<typename T>
void Test_TemplatedFunctionVariadic(int arg, ...);
// CHECK:      FunctionTemplateDecl{{.*}}Test_TemplatedFunctionVariadic
// CHECK:        ParamCommandComment{{.*}} [in] implicitly Param="..."
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" More arguments"
