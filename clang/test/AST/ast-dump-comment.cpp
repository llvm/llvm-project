// Without serialization:
// RUN: %clang_cc1 -Wdocumentation -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck -strict-whitespace %s
//
// With serialization:
// RUN: %clang_cc1 -Wdocumentation -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -Wdocumentation -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

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

/// \retval 42 Aaa
int Test_BlockCommandComment_WithArgs();
// CHECK:      FunctionDecl{{.*}}Test_BlockCommandComment_WithArgs
// CHECK:        BlockCommandComment{{.*}} Name="retval" Arg[0]="42"
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

/// \n Aaa
int Test_InlineCommandComment_NoArgs;
// CHECK:      VarDecl{{.*}}Test_InlineCommandComment_NoArgs
// CHECK:        InlineCommandComment{{.*}} Name="n" RenderNormal
// CHECK-NEXT:   TextComment{{.*}} Text=" Aaa"

/// \anchor Aaa
int Test_InlineCommandCommentAnchor;
// CHECK:      VarDecl{{.*}}Test_InlineCommandComment
// CHECK:        InlineCommandComment{{.*}} Name="anchor" RenderAnchor Arg[0]="Aaa"

/// \relates Aaa
int Test_InlineCommandCommentRelates;
// CHECK:      VarDecl{{.*}}Test_InlineCommandCommentRelates
// CHECK:        InlineCommandComment{{.*}} Name="relates" RenderNormal Arg[0]="Aaa"

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

/// <a
///     href="foo"
/// >Aaa</a>b
int Test_HTMLTagMultilineBCPL;
// CHECK:      VarDecl{{.*}}Test_HTMLTagMultilineBCPL
// CHECK-NEXT:   FullComment
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" "
// CHECK-NEXT:       HTMLStartTagComment{{.*}} Name="a" Attrs:  "href="foo"
// CHECK-NEXT:       TextComment{{.*}} Text="Aaa"
// CHECK-NEXT:       HTMLEndTagComment{{.*}} Name="a"
// CHECK-NEXT:       TextComment{{.*}} Text="b"

/// \verbatim
/// Aaa
/// \endverbatim
/// \f$ a \f$
/// \f( b \f)
/// \f[ c \f]
/// \f{env}{ c \f}
int Test_VerbatimBlockComment;
// CHECK:      VarDecl{{.*}}Test_VerbatimBlockComment
// CHECK:        VerbatimBlockComment{{.*}} Name="verbatim" CloseName="endverbatim"
// CHECK-NEXT:     VerbatimBlockLineComment{{.*}} Text=" Aaa"
// CHECK:        VerbatimBlockComment{{.*}} Name="f$" CloseName="f$"
// CHECK-NEXT:     VerbatimBlockLineComment{{.*}} Text=" a "
// CHECK:        VerbatimBlockComment{{.*}} Name="f(" CloseName="f)"
// CHECK-NEXT:     VerbatimBlockLineComment{{.*}} Text=" b "
// CHECK:        VerbatimBlockComment{{.*}} Name="f[" CloseName="f]"
// CHECK-NEXT:     VerbatimBlockLineComment{{.*}} Text=" c "
// CHECK:        VerbatimBlockComment{{.*}} Name="f{" CloseName="f}"
// CHECK-NEXT:     VerbatimBlockLineComment{{.*}} Text="env}{ c "

/// \param ... More arguments
template<typename T>
void Test_TemplatedFunctionVariadic(int arg, ...);
// CHECK:      FunctionTemplateDecl{{.*}}Test_TemplatedFunctionVariadic
// CHECK:        ParamCommandComment{{.*}} [in] implicitly Param="..."
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" More arguments"

/// \thread_safe test for underscore in special command
int Test_UnderscoreInSpecialCommand;
// CHECK:      VarDecl{{.*}}Test_UnderscoreInSpecialCommand 'int'
// CHECK:        InlineCommandComment{{.*}} Name="thread_safe" RenderNormal
// CHECK-NEXT:     TextComment{{.*}} Text=" test for underscore in special command"
