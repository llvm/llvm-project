// RUN: rm -rf %t
// RUN: mkdir %t

// This file contains UTF-8 sequences.  Please don't "fix" them!

// Check that we serialize comment source locations properly.
// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -o %t/out.pch %s
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t/out.pch -fsyntax-only %s

// RUN: c-index-test -write-pch %t/out.pch -x c++ -std=c++11 %s
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -std=c++11 > %t/out.c-index-direct
// RUN: c-index-test -test-load-tu %t/out.pch all > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-pch

// XFAIL: msan
// XFAIL: valgrind

#ifndef HEADER
#define HEADER

//===---
// Tests for \brief and its aliases.
//===---

/// Aaa.
void test_cmd_brief_like_1();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_1:{{.*}} BriefComment=[Aaa.] FullCommentAsHTML=[<p class="para-brief"> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_1</Name><USR>c:@F@test_cmd_brief_like_1#</USR><Declaration>void test_cmd_brief_like_1()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.])))]

/// \brief Aaa.
void test_cmd_brief_like_2();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_2:{{.*}} BriefComment=[Aaa.] FullCommentAsHTML=[<p class="para-brief"> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_2</Name><USR>c:@F@test_cmd_brief_like_2#</USR><Declaration>void test_cmd_brief_like_2()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

/// \short Aaa.
void test_cmd_brief_like_3();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_3:{{.*}} BriefComment=[Aaa.] FullCommentAsHTML=[<p class="para-brief"> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_3</Name><USR>c:@F@test_cmd_brief_like_3#</USR><Declaration>void test_cmd_brief_like_3()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[short]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

/// Aaa.
///
/// \brief Bbb.
void test_cmd_brief_like_4();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_4:{{.*}} BriefComment=[Bbb.] FullCommentAsHTML=[<p class="para-brief"> Bbb.</p><p> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_4</Name><USR>c:@F@test_cmd_brief_like_4#</USR><Declaration>void test_cmd_brief_like_4()</Declaration><Abstract><Para> Bbb.</Para></Abstract><Discussion><Para> Aaa.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
///
/// \brief Bbb.
///
/// Ccc.
void test_cmd_brief_like_5();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_5:{{.*}} BriefComment=[Bbb.] FullCommentAsHTML=[<p class="para-brief"> Bbb.</p><p> Aaa.</p><p> Ccc.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_5</Name><USR>c:@F@test_cmd_brief_like_5#</USR><Declaration>void test_cmd_brief_like_5()</Declaration><Abstract><Para> Bbb.</Para></Abstract><Discussion><Para> Aaa.</Para><Para> Ccc.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Ccc.])))]

/// \brief Aaa.
/// \brief Bbb.
void test_cmd_brief_like_6();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_6:{{.*}} BriefComment=[Bbb.] FullCommentAsHTML=[<p class="para-brief"> Aaa. </p><p class="para-brief"> Bbb.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_6</Name><USR>c:@F@test_cmd_brief_like_6#</USR><Declaration>void test_cmd_brief_like_6()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// \abstract Aaa.
///
/// Bbb.
void test_cmd_brief_like_7();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_brief_like_7:{{.*}} BriefComment=[Aaa.] FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p> Bbb.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_brief_like_7</Name><USR>c:@F@test_cmd_brief_like_7#</USR><Declaration>void test_cmd_brief_like_7()</Declaration><Abstract><Para> Aaa.</Para></Abstract><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[abstract]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Bbb.])))]

//===---
// Tests for \returns and its aliases.
//===---

/// Aaa.
///
/// \return Bbb.
void test_cmd_returns_like_1();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_returns_like_1:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><div class="result-discussion"><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p></div>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_returns_like_1</Name><USR>c:@F@test_cmd_returns_like_1#</USR><Declaration>void test_cmd_returns_like_1()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Bbb.</Para></ResultDiscussion></Function>]

// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[return]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
///
/// \returns Bbb.
void test_cmd_returns_like_2();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_returns_like_2:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><div class="result-discussion"><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p></div>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_returns_like_2</Name><USR>c:@F@test_cmd_returns_like_2#</USR><Declaration>void test_cmd_returns_like_2()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Bbb.</Para></ResultDiscussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
///
/// \result Bbb.
void test_cmd_returns_like_3();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_returns_like_3:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><div class="result-discussion"><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p></div>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_returns_like_3</Name><USR>c:@F@test_cmd_returns_like_3#</USR><Declaration>void test_cmd_returns_like_3()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Bbb.</Para></ResultDiscussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[result]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// \returns Aaa.
/// \returns Bbb.
void test_cmd_returns_like_4();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_returns_like_4:{{.*}} FullCommentAsHTML=[<div class="result-discussion"><p class="para-returns"><span class="word-returns">Returns</span>  Aaa. </p><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p></div>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_returns_like_4</Name><USR>c:@F@test_cmd_returns_like_4#</USR><Declaration>void test_cmd_returns_like_4()</Declaration><ResultDiscussion><Para> Aaa. </Para><Para> Bbb.</Para></ResultDiscussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
///
/// Bbb.
///
/// \returns Ccc.
void test_cmd_returns_like_5();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_returns_like_5:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p> Bbb.</p><div class="result-discussion"><p class="para-returns"><span class="word-returns">Returns</span>  Ccc.</p></div>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_returns_like_5</Name><USR>c:@F@test_cmd_returns_like_5#</USR><Declaration>void test_cmd_returns_like_5()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Ccc.</Para></ResultDiscussion><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Bbb.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc.]))))]

//===---
// Tests for \param.
//===---

/// \param
void test_cmd_param_1(int x1);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_param_1:{{.*}} FullCommentAsHTML=[] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_param_1</Name><USR>c:@F@test_cmd_param_1#I#</USR><Declaration>void test_cmd_param_1(int x1)</Declaration></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph IsWhitespace)))]

/// \param x1 Aaa.
void test_cmd_param_2(int x1);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_param_2:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_param_2</Name><USR>c:@F@test_cmd_param_2#I#</USR><Declaration>void test_cmd_param_2(int x1)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

/// \param zzz Aaa.
void test_cmd_param_3(int x1);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_param_3:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-invalid">zzz</dt><dd class="param-descr-index-invalid"> Aaa.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_param_3</Name><USR>c:@F@test_cmd_param_3#I#</USR><Declaration>void test_cmd_param_3(int x1)</Declaration><Parameters><Parameter><Name>zzz</Name><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[zzz] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

/// \param x2 Bbb.
/// \param x1 Aaa.
void test_cmd_param_4(int x1, int x2);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_param_4:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa.</dd><dt class="param-name-index-1">x2</dt><dd class="param-descr-index-1"> Bbb. </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_param_4</Name><USR>c:@F@test_cmd_param_4#I#I#</USR><Declaration>void test_cmd_param_4(int x1, int x2)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter><Parameter><Name>x2</Name><Index>1</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Bbb. </Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x2] ParamIndex=1
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

/// \param x2 Bbb.
/// \param zzz Aaa.
/// \param x1 Aaa.
void test_cmd_param_5(int x1, int x2);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_param_5:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa.</dd><dt class="param-name-index-1">x2</dt><dd class="param-descr-index-1"> Bbb. </dd><dt class="param-name-index-invalid">zzz</dt><dd class="param-descr-index-invalid"> Aaa. </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_param_5</Name><USR>c:@F@test_cmd_param_5#I#I#</USR><Declaration>void test_cmd_param_5(int x1, int x2)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter><Parameter><Name>x2</Name><Index>1</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Bbb. </Para></Discussion></Parameter><Parameter><Name>zzz</Name><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa. </Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x2] ParamIndex=1
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[zzz] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

/// \param x1 Aaa.
/// \param ... Bbb.
void test_cmd_param_6(int x1, ...);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_cmd_param_6:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa. </dd><dt class="param-name-index-vararg">...</dt><dd class="param-descr-index-vararg"> Bbb.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_param_6</Name><USR>c:@F@test_cmd_param_6#I.#</USR><Declaration>void test_cmd_param_6(int x1, ...)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa. </Para></Discussion></Parameter><Parameter><Name>...</Name><IsVarArg /><Direction isExplicit="0">in</Direction><Discussion><Para> Bbb.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[...] ParamIndex=4294967295
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

//===---
// Tests for \tparam.
//===---

/// \tparam
/// \param aaa Blah blah
template<typename T>
void test_cmd_tparam_1(T aaa);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionTemplate=test_cmd_tparam_1:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">aaa</dt><dd class="param-descr-index-0"> Blah blah</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_tparam_1</Name><USR>c:@FT@&gt;1#Ttest_cmd_tparam_1#t0.0#v#</USR><Declaration>template &lt;typename T&gt; void test_cmd_tparam_1(T aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[] ParamPosition=Invalid
// CHECK-NEXT:         (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[aaa] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Blah blah]))))]

/// \tparam T
/// \param aaa Blah blah
template<typename T>
void test_cmd_tparam_2(T aaa);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionTemplate=test_cmd_tparam_2:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">aaa</dt><dd class="param-descr-index-0"> Blah blah</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_tparam_2</Name><USR>c:@FT@&gt;1#Ttest_cmd_tparam_2#t0.0#v#</USR><Declaration>template &lt;typename T&gt; void test_cmd_tparam_2(T aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[aaa] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Blah blah]))))]

/// \tparam T2 Bbb
/// \tparam T1 Aaa
template<typename T1, typename T2>
void test_cmd_tparam_3(T1 aaa, T2 bbb);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionTemplate=test_cmd_tparam_3:{{.*}} FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">T1</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">T2</dt><dd class="tparam-descr-index-1"> Bbb </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_tparam_3</Name><USR>c:@FT@&gt;2#T#Ttest_cmd_tparam_3#t0.0#t0.1#v#</USR><Declaration>template &lt;typename T1, typename T2&gt; void test_cmd_tparam_3(T1 aaa, T2 bbb)</Declaration><TemplateParameters><Parameter><Name>T1</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>T2</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter></TemplateParameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T2] ParamPosition={1}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T1] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa]))))]

/// \tparam T2 Bbb
/// \tparam U Zzz
/// \tparam V Ccc
/// \tparam T1 Aaa
template<typename T1, typename T2, int V>
void test_cmd_tparam_4(T1 aaa, T2 bbb);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionTemplate=test_cmd_tparam_4:{{.*}} FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">T1</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">T2</dt><dd class="tparam-descr-index-1"> Bbb </dd><dt class="tparam-name-index-2">V</dt><dd class="tparam-descr-index-2"> Ccc </dd><dt class="tparam-name-index-invalid">U</dt><dd class="tparam-descr-index-invalid"> Zzz </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_tparam_4</Name><USR>c:@FT@&gt;3#T#T#NItest_cmd_tparam_4#t0.0#t0.1#v#</USR><Declaration>template &lt;typename T1, typename T2, int V&gt;\nvoid test_cmd_tparam_4(T1 aaa, T2 bbb)</Declaration><TemplateParameters><Parameter><Name>T1</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>T2</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>V</Name><Index>2</Index><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>U</Name><Discussion><Para> Zzz </Para></Discussion></Parameter></TemplateParameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T2] ParamPosition={1}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[U] ParamPosition=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Zzz] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[V] ParamPosition={2}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T1] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa]))))]

/// \tparam TTT Ddd
/// \tparam C Ccc
/// \tparam T Aaa
/// \tparam TT Bbb
template<template<template<typename T> class TT, class C> class TTT>
void test_cmd_tparam_5();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionTemplate=test_cmd_tparam_5:{{.*}} FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">TTT</dt><dd class="tparam-descr-index-0"> Ddd </dd><dt class="tparam-name-index-other">C</dt><dd class="tparam-descr-index-other"> Ccc </dd><dt class="tparam-name-index-other">T</dt><dd class="tparam-descr-index-other"> Aaa </dd><dt class="tparam-name-index-other">TT</dt><dd class="tparam-descr-index-other"> Bbb</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_cmd_tparam_5</Name><USR>c:@FT@&gt;1#t&gt;2#t&gt;1#T#Ttest_cmd_tparam_5#v#</USR><Declaration>template &lt;template &lt;template &lt;typename T&gt; class TT, class C&gt; class TTT&gt;\nvoid test_cmd_tparam_5()</Declaration><TemplateParameters><Parameter><Name>TTT</Name><Index>0</Index><Discussion><Para> Ddd </Para></Discussion></Parameter><Parameter><Name>C</Name><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>T</Name><Discussion><Para> Aaa </Para></Discussion></Parameter><Parameter><Name>TT</Name><Discussion><Para> Bbb</Para></Discussion></Parameter></TemplateParameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[TTT] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ddd] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[C] ParamPosition={0, 1}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T] ParamPosition={0, 0, 0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[TT] ParamPosition={0, 0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb]))))]

//===---
// Tests for interaction between commands.
//===---

/// \brief Aaa.
///
/// Bbb.
///
/// \param x2 Ddd.
/// \param x1 Ccc.
/// \returns Eee.
void test_full_comment_1(int x1, int x2);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=test_full_comment_1:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p> Bbb.</p><dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Ccc. </dd><dt class="param-name-index-1">x2</dt><dd class="param-descr-index-1"> Ddd. </dd></dl><div class="result-discussion"><p class="para-returns"><span class="word-returns">Returns</span>  Eee.</p></div>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>test_full_comment_1</Name><USR>c:@F@test_full_comment_1#I#I#</USR><Declaration>void test_full_comment_1(int x1, int x2)</Declaration><Abstract><Para> Aaa.</Para></Abstract><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Ccc. </Para></Discussion></Parameter><Parameter><Name>x2</Name><Index>1</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Ddd. </Para></Discussion></Parameter></Parameters><ResultDiscussion><Para> Eee.</Para></ResultDiscussion><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Bbb.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x2] ParamIndex=1
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ddd.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Eee.]))))]

//===---
// Misc tests.
//===---

/// <br><a href="http://example.com/">Aaa</a>
void comment_to_html_conversion_24();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_24:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <br><a href="http://example.com/">Aaa</a></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_24</Name><USR>c:@F@comment_to_html_conversion_24#</USR><Declaration>void comment_to_html_conversion_24()</Declaration><Abstract><Para> <rawHTML><![CDATA[<br>]]></rawHTML><rawHTML><![CDATA[<a href="http://example.com/">]]></rawHTML>Aaa<rawHTML>&lt;/a&gt;</rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[br])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[a] Attrs: href=http://example.com/)
// CHECK-NEXT:         (CXComment_Text Text=[Aaa])
// CHECK-NEXT:         (CXComment_HTMLEndTag Name=[a])))]

/// \verbatim
/// <a href="http://example.com/">Aaa</a>
/// <a href='http://example.com/'>Aaa</a>
/// \endverbatim
void comment_to_html_conversion_25();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_25:{{.*}} FullCommentAsHTML=[<pre> &lt;a href=&quot;http:&#47;&#47;example.com&#47;&quot;&gt;Aaa&lt;&#47;a&gt;\n &lt;a href=&#39;http:&#47;&#47;example.com&#47;&#39;&gt;Aaa&lt;&#47;a&gt;</pre>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_25</Name><USR>c:@F@comment_to_html_conversion_25#</USR><Declaration>void comment_to_html_conversion_25()</Declaration><Discussion><Verbatim xml:space="preserve" kind="verbatim"> &lt;a href=&quot;http://example.com/&quot;&gt;Aaa&lt;/a&gt;\n &lt;a href=&apos;http://example.com/&apos;&gt;Aaa&lt;/a&gt;</Verbatim></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimBlockCommand CommandName=[verbatim]
// CHECK-NEXT:         (CXComment_VerbatimBlockLine Text=[ <a href="http://example.com/">Aaa</a>])
// CHECK-NEXT:         (CXComment_VerbatimBlockLine Text=[ <a href='http://example.com/'>Aaa</a>])))]

/// \def foo_def
/// \fn foo_fn
/// \namespace foo_namespace
/// \overload foo_overload
/// \property foo_property
/// \typedef foo_typedef
/// \var foo_var
/// \function foo_function
/// \class foo_class
/// \method foo_method
/// \interface foo_interface
/// Blah blah.
void comment_to_html_conversion_26();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_26:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Blah blah.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_26</Name><USR>c:@F@comment_to_html_conversion_26#</USR><Declaration>void comment_to_html_conversion_26()</Declaration><Abstract><Para> Blah blah.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_def])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_fn])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_namespace])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_overload])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_property])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_typedef])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_var])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_function])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_class])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_method])
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimLine Text=[ foo_interface])
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Blah blah.])))]

/// \unknown
void comment_to_html_conversion_27();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_27:{{.*}} FullCommentAsHTML=[<p class="para-brief"> </p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_27</Name><USR>c:@F@comment_to_html_conversion_27#</USR><Declaration>void comment_to_html_conversion_27()</Declaration><Abstract><Para> </Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[unknown] RenderNormal)))]

/// \b Aaa
void comment_to_html_conversion_28();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_28:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <b>Aaa</b></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_28</Name><USR>c:@F@comment_to_html_conversion_28#</USR><Declaration>void comment_to_html_conversion_28()</Declaration><Abstract><Para> <bold>Aaa</bold></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[b] RenderBold Arg[0]=Aaa)))]

/// \c Aaa \p Bbb
void comment_to_html_conversion_29();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_29:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <tt>Aaa</tt> <tt>Bbb</tt></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_29</Name><USR>c:@F@comment_to_html_conversion_29#</USR><Declaration>void comment_to_html_conversion_29()</Declaration><Abstract><Para> <monospaced>Aaa</monospaced> <monospaced>Bbb</monospaced></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[c] RenderMonospaced Arg[0]=Aaa)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[p] RenderMonospaced Arg[0]=Bbb)))]

/// \a Aaa \e Bbb \em Ccc
void comment_to_html_conversion_30();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_30:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <em>Aaa</em> <em>Bbb</em> <em>Ccc</em></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_30</Name><USR>c:@F@comment_to_html_conversion_30#</USR><Declaration>void comment_to_html_conversion_30()</Declaration><Abstract><Para> <emphasized>Aaa</emphasized> <emphasized>Bbb</emphasized> <emphasized>Ccc</emphasized></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[a] RenderEmphasized Arg[0]=Aaa)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[e] RenderEmphasized Arg[0]=Bbb)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[em] RenderEmphasized Arg[0]=Ccc)))]

/// \a 1<2 \e 3<4 \em 5<6 \param 7<8 aaa \tparam 9<10 bbb
void comment_to_html_conversion_31();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_31:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <em>1&lt;2</em> <em>3&lt;4</em> <em>5&lt;6</em> </p><dl><dt class="tparam-name-index-invalid">9&lt;10</dt><dd class="tparam-descr-index-invalid"> bbb</dd></dl><dl><dt class="param-name-index-invalid">7&lt;8</dt><dd class="param-descr-index-invalid"> aaa </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_31</Name><USR>c:@F@comment_to_html_conversion_31#</USR><Declaration>void comment_to_html_conversion_31()</Declaration><Abstract><Para> <emphasized>1&lt;2</emphasized> <emphasized>3&lt;4</emphasized> <emphasized>5&lt;6</emphasized> </Para></Abstract><TemplateParameters><Parameter><Name>9&lt;10</Name><Discussion><Para> bbb</Para></Discussion></Parameter></TemplateParameters><Parameters><Parameter><Name>7&lt;8</Name><Direction isExplicit="0">in</Direction><Discussion><Para> aaa </Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[a] RenderEmphasized Arg[0]=1<2)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[e] RenderEmphasized Arg[0]=3<4)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[em] RenderEmphasized Arg[0]=5<6)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[7<8] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ aaa ])))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[9<10] ParamPosition=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ bbb]))))]

/// \\ \@ \& \$ \# \< \> \% \" \. \::
void comment_to_html_conversion_32();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_32:{{.*}} FullCommentAsHTML=[<p class="para-brief"> \ @ &amp; $ # &lt; &gt; % &quot; . ::</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_32</Name><USR>c:@F@comment_to_html_conversion_32#</USR><Declaration>void comment_to_html_conversion_32()</Declaration><Abstract><Para> \ @ &amp; $ # &lt; &gt; % &quot; . ::</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[\])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[@])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[&])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[$])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[#])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[<])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[>])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[%])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=["])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[.])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[::])))]

/// &amp; &lt; &gt; &quot; &apos; &#109;&#101;&#111;&#119; &#x6d;&#x65;&#x6F;&#X77;
void comment_to_html_conversion_33();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_33:{{.*}} FullCommentAsHTML=[<p class="para-brief"> &amp; &lt; &gt; &quot; &#39; meow meow</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_33</Name><USR>c:@F@comment_to_html_conversion_33#</USR><Declaration>void comment_to_html_conversion_33()</Declaration><Abstract><Para> &amp; &lt; &gt; &quot; &apos; meow  meow</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[&])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[<])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[>])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=["])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=['])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[m])
// CHECK-NEXT:         (CXComment_Text Text=[e])
// CHECK-NEXT:         (CXComment_Text Text=[o])
// CHECK-NEXT:         (CXComment_Text Text=[w])
// CHECK-NEXT:         (CXComment_Text Text=[  ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[m])
// CHECK-NEXT:         (CXComment_Text Text=[e])
// CHECK-NEXT:         (CXComment_Text Text=[o])
// CHECK-NEXT:         (CXComment_Text Text=[w])))]

/// <em>0&lt;i</em>
void comment_to_html_conversion_34();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_34:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <em>0&lt;i</em></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_34</Name><USR>c:@F@comment_to_html_conversion_34#</USR><Declaration>void comment_to_html_conversion_34()</Declaration><Abstract><Para> <rawHTML><![CDATA[<em>]]></rawHTML>0&lt;i<rawHTML>&lt;/em&gt;</rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[em])
// CHECK-NEXT:         (CXComment_Text Text=[0])
// CHECK-NEXT:         (CXComment_Text Text=[<])
// CHECK-NEXT:         (CXComment_Text Text=[i])
// CHECK-NEXT:         (CXComment_HTMLEndTag Name=[em])))]

/// &copy; the copyright symbol
/// &trade; the trade mark symbol
/// &reg; the registered trade mark symbol
/// &nbsp; a non-breakable space.
/// &Delta; Greek letter Delta Δ.
/// &Gamma; Greek letter Gamma Γ.
void comment_to_html_conversion_35();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_35:{{.*}} FullCommentAsHTML=[<p class="para-brief"> © the copyright symbol ™ the trade mark symbol ® the registered trade mark symbol   a non-breakable space. Δ Greek letter Delta Δ. Γ Greek letter Gamma Γ.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_35</Name><USR>c:@F@comment_to_html_conversion_35#</USR><Declaration>void comment_to_html_conversion_35()</Declaration><Abstract><Para> © the copyright symbol ™ the trade mark symbol ® the registered trade mark symbol   a non-breakable space. Δ Greek letter Delta Δ. Γ Greek letter Gamma Γ.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[©])
// CHECK-NEXT:         (CXComment_Text Text=[ the copyright symbol] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[™])
// CHECK-NEXT:         (CXComment_Text Text=[ the trade mark symbol] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[®])
// CHECK-NEXT:         (CXComment_Text Text=[ the registered trade mark symbol] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[ ])
// CHECK-NEXT:         (CXComment_Text Text=[ a non-breakable space.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[Δ])
// CHECK-NEXT:         (CXComment_Text Text=[ Greek letter Delta Δ.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[Γ])
// CHECK-NEXT:         (CXComment_Text Text=[ Greek letter Gamma Γ.])))]

/// <h1 id="]]>">Aaa</h1>
void comment_to_html_conversion_36();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_36:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <h1 id="]]>">Aaa</h1></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_36</Name><USR>c:@F@comment_to_html_conversion_36#</USR><Declaration>void comment_to_html_conversion_36()</Declaration><Abstract><Para> <rawHTML><![CDATA[<h1 id="]]]]><![CDATA[>">]]></rawHTML>Aaa<rawHTML>&lt;/h1&gt;</rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[h1] Attrs: id=]]>)
// CHECK-NEXT:         (CXComment_Text Text=[Aaa])
// CHECK-NEXT:         (CXComment_HTMLEndTag Name=[h1])))]

/// \anchor A
void comment_to_html_conversion_37();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_37:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <span id="A"></span></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_37</Name><USR>c:@F@comment_to_html_conversion_37#</USR><Declaration>void comment_to_html_conversion_37()</Declaration><Abstract><Para> <anchor id="A"></anchor></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[anchor] RenderAnchor Arg[0]=A)))]

/// Aaa bbb<img/>
void comment_to_html_conversion_38();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_38:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa bbb<img/></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_38</Name><USR>c:@F@comment_to_html_conversion_38#</USR><Declaration>void comment_to_html_conversion_38()</Declaration><Abstract><Para> Aaa bbb<rawHTML><![CDATA[<img/>]]></rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa bbb])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[img] SelfClosing)

/// Aaa ccc<img />
void comment_to_html_conversion_39();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_39:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa ccc<img/></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_39</Name><USR>c:@F@comment_to_html_conversion_39#</USR><Declaration>void comment_to_html_conversion_39()</Declaration><Abstract><Para> Aaa ccc<rawHTML><![CDATA[<img/>]]></rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa ccc])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[img] SelfClosing)

/// Aaa ccc<img src="">
void comment_to_html_conversion_40();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_40:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa ccc<img src></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_40</Name><USR>c:@F@comment_to_html_conversion_40#</USR><Declaration>void comment_to_html_conversion_40()</Declaration><Abstract><Para> Aaa ccc<rawHTML><![CDATA[<img src>]]></rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa ccc])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[img] Attrs: src=)

/// Aaa ccc<img src="path">
void comment_to_html_conversion_41();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_41:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa ccc<img src="path"></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_41</Name><USR>c:@F@comment_to_html_conversion_41#</USR><Declaration>void comment_to_html_conversion_41()</Declaration><Abstract><Para> Aaa ccc<rawHTML><![CDATA[<img src="path">]]></rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa ccc])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[img] Attrs: src=path)

/// Aaa ccc<img src="path"/>
void comment_to_html_conversion_42();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_42:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa ccc<img src="path"/></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_42</Name><USR>c:@F@comment_to_html_conversion_42#</USR><Declaration>void comment_to_html_conversion_42()</Declaration><Abstract><Para> Aaa ccc<rawHTML><![CDATA[<img src="path"/>]]></rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa ccc])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[img] Attrs: src=path SelfClosing)

/// Aaa ddd<img src=""/>
void comment_to_html_conversion_43();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_html_conversion_43:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa ddd<img src/></p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_html_conversion_43</Name><USR>c:@F@comment_to_html_conversion_43#</USR><Declaration>void comment_to_html_conversion_43()</Declaration><Abstract><Para> Aaa ddd<rawHTML><![CDATA[<img src/>]]></rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa ddd])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[img] Attrs: src= SelfClosing)

/// Aaa.
class comment_to_xml_conversion_01 {
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:7: ClassDecl=comment_to_xml_conversion_01:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="7"><Name>comment_to_xml_conversion_01</Name><USR>c:@S@comment_to_xml_conversion_01</USR><Declaration>class comment_to_xml_conversion_01 {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]

  /// \param aaa Blah blah.
  comment_to_xml_conversion_01(int aaa);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:3: CXXConstructor=comment_to_xml_conversion_01:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="3"><Name>comment_to_xml_conversion_01</Name><USR>c:@S@comment_to_xml_conversion_01@F@comment_to_xml_conversion_01#I#</USR><Declaration>comment_to_xml_conversion_01(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]

  /// Aaa.
  ~comment_to_xml_conversion_01();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:3: CXXDestructor=~comment_to_xml_conversion_01:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="3"><Name>~comment_to_xml_conversion_01</Name><USR>c:@S@comment_to_xml_conversion_01@F@~comment_to_xml_conversion_01#</USR><Declaration>~comment_to_xml_conversion_01()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]

  /// \param aaa Blah blah.
  int comment_to_xml_conversion_02(int aaa);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:7: CXXMethod=comment_to_xml_conversion_02:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="7"><Name>comment_to_xml_conversion_02</Name><USR>c:@S@comment_to_xml_conversion_01@F@comment_to_xml_conversion_02#I#</USR><Declaration>int comment_to_xml_conversion_02(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]

  /// \param aaa Blah blah.
  static int comment_to_xml_conversion_03(int aaa);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:14: CXXMethod=comment_to_xml_conversion_03:{{.*}} FullCommentAsXML=[<Function isClassMethod="1" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="14"><Name>comment_to_xml_conversion_03</Name><USR>c:@S@comment_to_xml_conversion_01@F@comment_to_xml_conversion_03#I#S</USR><Declaration>static int comment_to_xml_conversion_03(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]

  /// Aaa.
  int comment_to_xml_conversion_04;

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:7: FieldDecl=comment_to_xml_conversion_04:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="7"><Name>comment_to_xml_conversion_04</Name><USR>c:@S@comment_to_xml_conversion_01@FI@comment_to_xml_conversion_04</USR><Declaration>int comment_to_xml_conversion_04</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]

  /// Aaa.
  static int comment_to_xml_conversion_05;

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:14: VarDecl=comment_to_xml_conversion_05:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="14"><Name>comment_to_xml_conversion_05</Name><USR>c:@S@comment_to_xml_conversion_01@comment_to_xml_conversion_05</USR><Declaration>static int comment_to_xml_conversion_05</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]

  /// \param aaa Blah blah.
  void operator()(int aaa);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:8: CXXMethod=operator():{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="8"><Name>operator()</Name><USR>c:@S@comment_to_xml_conversion_01@F@operator()#I#</USR><Declaration>void operator()(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]

  /// Aaa.
  operator bool();

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:3: CXXConversion=operator bool:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="3"><Name>operator bool</Name><USR>c:@S@comment_to_xml_conversion_01@F@operator bool#</USR><Declaration>operator bool()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]

  /// Aaa.
  typedef int comment_to_xml_conversion_06;

// USR is line-dependent here, so filter it with a regexp.
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-3]]:15: TypedefDecl=comment_to_xml_conversion_06:{{.*}} FullCommentAsXML=[<Typedef file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-3]]" column="15"><Name>comment_to_xml_conversion_06</Name><USR>{{[^<]+}}</USR><Declaration>typedef int comment_to_xml_conversion_06</Declaration><Abstract><Para> Aaa.</Para></Abstract></Typedef>]

  /// Aaa.
  using comment_to_xml_conversion_07 = int;

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:9: TypeAliasDecl=comment_to_xml_conversion_07:{{.*}} FullCommentAsXML=[<Typedef file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="9"><Name>comment_to_xml_conversion_07</Name><USR>c:@S@comment_to_xml_conversion_01@comment_to_xml_conversion_07</USR><Declaration>using comment_to_xml_conversion_07 = int</Declaration><Abstract><Para> Aaa.</Para></Abstract></Typedef>]

  /// Aaa.
  template<typename T, typename U>
  class comment_to_xml_conversion_08 { };

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:9: ClassTemplate=comment_to_xml_conversion_08:{{.*}} FullCommentAsXML=[<Class templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="9"><Name>comment_to_xml_conversion_08</Name><USR>c:@S@comment_to_xml_conversion_01@ST&gt;2#T#T@comment_to_xml_conversion_08</USR><Declaration>template &lt;typename T, typename U&gt; class comment_to_xml_conversion_08 {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]

  /// Aaa.
  template<typename T>
  using comment_to_xml_conversion_09 = comment_to_xml_conversion_08<T, int>;

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:3: TypeAliasTemplateDecl=comment_to_xml_conversion_09:{{.*}} FullCommentAsXML=[<Typedef file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="3"><Name>comment_to_xml_conversion_09</Name><USR>c:@S@comment_to_xml_conversion_01@comment_to_xml_conversion_09</USR><Declaration>template &lt;typename T&gt;\nusing comment_to_xml_conversion_09 = comment_to_xml_conversion_08&lt;T, int&gt;</Declaration><Abstract><Para> Aaa.</Para></Abstract></Typedef>]
};

/// Aaa.
template<typename T, typename U>
void comment_to_xml_conversion_10(T aaa, U bbb);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionTemplate=comment_to_xml_conversion_10:{{.*}} FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_xml_conversion_10</Name><USR>c:@FT@&gt;2#T#Tcomment_to_xml_conversion_10#t0.0#t0.1#v#</USR><Declaration>template &lt;typename T, typename U&gt;\nvoid comment_to_xml_conversion_10(T aaa, U bbb)</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]

/// Aaa.
template<>
void comment_to_xml_conversion_10(int aaa, int bbb);

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:6: FunctionDecl=comment_to_xml_conversion_10:{{.*}} FullCommentAsXML=[<Function templateKind="specialization" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="6"><Name>comment_to_xml_conversion_10</Name><USR>c:@F@comment_to_xml_conversion_10&lt;#I#I&gt;#I#I#</USR><Declaration>template &lt;&gt; void comment_to_xml_conversion_10&lt;int, int&gt;(int aaa, int bbb)</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]

/// Aaa.
template<typename T, typename U>
class comment_to_xml_conversion_11 { };

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:7: ClassTemplate=comment_to_xml_conversion_11:{{.*}} FullCommentAsXML=[<Class templateKind="template" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="7"><Name>comment_to_xml_conversion_11</Name><USR>c:@ST&gt;2#T#T@comment_to_xml_conversion_11</USR><Declaration>template &lt;typename T, typename U&gt; class comment_to_xml_conversion_11 {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]

/// Aaa.
template<typename T>
class comment_to_xml_conversion_11<T, int> { };

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:7: ClassTemplatePartialSpecialization=comment_to_xml_conversion_11:{{.*}} FullCommentAsXML=[<Class templateKind="partialSpecialization" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="7"><Name>comment_to_xml_conversion_11</Name><USR>c:@SP&gt;1#T@comment_to_xml_conversion_11&gt;#t0.0#I</USR><Declaration>template &lt;typename T&gt; class comment_to_xml_conversion_11&lt;T, int&gt; {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]

/// Aaa.
template<>
class comment_to_xml_conversion_11<int, int> { };

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:7: ClassDecl=comment_to_xml_conversion_11:{{.*}} FullCommentAsXML=[<Class templateKind="specialization" file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="7"><Name>comment_to_xml_conversion_11</Name><USR>c:@S@comment_to_xml_conversion_11&gt;#I#I</USR><Declaration>template &lt;&gt; class comment_to_xml_conversion_11&lt;int, int&gt; {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]

/// Aaa.
int comment_to_xml_conversion_12;

// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-2]]:5: VarDecl=comment_to_xml_conversion_12:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-2]]" column="5"><Name>comment_to_xml_conversion_12</Name><USR>c:@comment_to_xml_conversion_12</USR><Declaration>int comment_to_xml_conversion_12</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]

/// Aaa.
namespace comment_to_xml_conversion_13 {
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:11: Namespace=comment_to_xml_conversion_13:{{.*}} FullCommentAsXML=[<Namespace file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="11"><Name>comment_to_xml_conversion_13</Name><USR>c:@N@comment_to_xml_conversion_13</USR><Declaration>namespace comment_to_xml_conversion_13 {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Namespace>]

  /// Aaa.
  namespace comment_to_xml_conversion_14 {
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:13: Namespace=comment_to_xml_conversion_14:{{.*}} FullCommentAsXML=[<Namespace file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="13"><Name>comment_to_xml_conversion_14</Name><USR>c:@N@comment_to_xml_conversion_13@N@comment_to_xml_conversion_14</USR><Declaration>namespace comment_to_xml_conversion_14 {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Namespace>]
  }
}

/// Aaa.
enum comment_to_xml_conversion_15 {
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: EnumDecl=comment_to_xml_conversion_15:{{.*}} FullCommentAsXML=[<Enum file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_15</Name><USR>c:@E@comment_to_xml_conversion_15</USR><Declaration>enum comment_to_xml_conversion_15{{( : int)?}} {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Enum>]

  /// Aaa.
  comment_to_xml_conversion_16
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:3: EnumConstantDecl=comment_to_xml_conversion_16:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="3"><Name>comment_to_xml_conversion_16</Name><USR>c:@E@comment_to_xml_conversion_15@comment_to_xml_conversion_16</USR><Declaration>comment_to_xml_conversion_16</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
};

/// Aaa.
enum class comment_to_xml_conversion_17 {
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:12: EnumDecl=comment_to_xml_conversion_17:{{.*}} FullCommentAsXML=[<Enum file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="12"><Name>comment_to_xml_conversion_17</Name><USR>c:@E@comment_to_xml_conversion_17</USR><Declaration>enum class comment_to_xml_conversion_17 : int {}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Enum>]

  /// Aaa.
  comment_to_xml_conversion_18
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:3: EnumConstantDecl=comment_to_xml_conversion_18:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="3"><Name>comment_to_xml_conversion_18</Name><USR>c:@E@comment_to_xml_conversion_17@comment_to_xml_conversion_18</USR><Declaration>comment_to_xml_conversion_18</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
};

/// <a href="http://example.org/">
void comment_to_xml_conversion_unsafe_html_01();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_01:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_01</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_01#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_01()</Declaration><Abstract><Para> <rawHTML isMalformed="1"><![CDATA[<a href="http://example.org/">]]></rawHTML></Para></Abstract></Function>]

/// <a href="http://example.org/"><em>Aaa</em>
void comment_to_xml_conversion_unsafe_html_02();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_02:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_02</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_02#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_02()</Declaration><Abstract><Para> <rawHTML isMalformed="1"><![CDATA[<a href="http://example.org/">]]></rawHTML><rawHTML><![CDATA[<em>]]></rawHTML>Aaa<rawHTML>&lt;/em&gt;</rawHTML></Para></Abstract></Function>]

/// <em>Aaa
void comment_to_xml_conversion_unsafe_html_03();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_03:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_03</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_03#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_03()</Declaration><Abstract><Para> <rawHTML isMalformed="1"><![CDATA[<em>]]></rawHTML>Aaa</Para></Abstract></Function>]

/// <em>Aaa</b></em>
void comment_to_xml_conversion_unsafe_html_04();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_04:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_04</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_04#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_04()</Declaration><Abstract><Para> <rawHTML><![CDATA[<em>]]></rawHTML>Aaa<rawHTML isMalformed="1">&lt;/b&gt;</rawHTML><rawHTML>&lt;/em&gt;</rawHTML></Para></Abstract></Function>]

/// <em>Aaa</em></b>
void comment_to_xml_conversion_unsafe_html_05();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_05:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_05</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_05#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_05()</Declaration><Abstract><Para> <rawHTML><![CDATA[<em>]]></rawHTML>Aaa<rawHTML>&lt;/em&gt;</rawHTML><rawHTML isMalformed="1">&lt;/b&gt;</rawHTML></Para></Abstract></Function>]

/// </table>
void comment_to_xml_conversion_unsafe_html_06();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_06:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_06</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_06#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_06()</Declaration><Abstract><Para> <rawHTML isMalformed="1">&lt;/table&gt;</rawHTML></Para></Abstract></Function>]

/// <div onclick="alert('meow');">Aaa</div>
void comment_to_xml_conversion_unsafe_html_07();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_unsafe_html_07:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_unsafe_html_07</Name><USR>c:@F@comment_to_xml_conversion_unsafe_html_07#</USR><Declaration>void comment_to_xml_conversion_unsafe_html_07()</Declaration><Abstract><Para> <rawHTML><![CDATA[<div onclick="alert('meow');">]]></rawHTML>Aaa<rawHTML>&lt;/div&gt;</rawHTML></Para></Abstract></Function>]

//===---
// Check that we attach comments from the base class to derived classes if they don't have a comment.
//===---

/// BaseToSuper1_Base
class BaseToSuper1_Base {};

class BaseToSuper1_Derived : public BaseToSuper1_Base {};
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:7: ClassDecl=BaseToSuper1_Derived:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="7"><Name>BaseToSuper1_Base</Name><USR>c:@S@BaseToSuper1_Base</USR><Declaration>class BaseToSuper1_Derived : public BaseToSuper1_Base {}</Declaration><Abstract><Para> BaseToSuper1_Base</Para></Abstract></Class>]


/// BaseToSuper2_Base
class BaseToSuper2_Base {};

/// BaseToSuper2_Derived
class BaseToSuper2_Derived : public BaseToSuper2_Base {};
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:7: ClassDecl=BaseToSuper2_Derived:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="7"><Name>BaseToSuper2_Derived</Name><USR>c:@S@BaseToSuper2_Derived</USR><Declaration>class BaseToSuper2_Derived : public BaseToSuper2_Base {}</Declaration><Abstract><Para> BaseToSuper2_Derived</Para></Abstract></Class>]

class BaseToSuper2_MoreDerived : public BaseToSuper2_Derived {};
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:7: ClassDecl=BaseToSuper2_MoreDerived:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="7"><Name>BaseToSuper2_Derived</Name><USR>c:@S@BaseToSuper2_Derived</USR><Declaration>class BaseToSuper2_MoreDerived : public BaseToSuper2_Derived {}</Declaration><Abstract><Para> BaseToSuper2_Derived</Para></Abstract></Class>]


/// BaseToSuper3_Base
class BaseToSuper3_Base {};

class BaseToSuper3_DerivedA : public virtual BaseToSuper3_Base {};

class BaseToSuper3_DerivedB : public virtual BaseToSuper3_Base {};

class BaseToSuper3_MoreDerived : public BaseToSuper3_DerivedA, public BaseToSuper3_DerivedB {};
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:7: ClassDecl=BaseToSuper3_MoreDerived:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="7"><Name>BaseToSuper3_Base</Name><USR>c:@S@BaseToSuper3_Base</USR><Declaration>class BaseToSuper3_MoreDerived : public BaseToSuper3_DerivedA,\n                                 public BaseToSuper3_DerivedB {}</Declaration><Abstract><Para> BaseToSuper3_Base</Para></Abstract></Class>]


// Check that we propagate comments only through public inheritance.

/// BaseToSuper4_Base
class BaseToSuper4_Base {};

/// BaseToSuper4_DerivedA
class BaseToSuper4_DerivedA : virtual BaseToSuper4_Base {};

class BaseToSuper4_DerivedB : public virtual BaseToSuper4_Base {};

class BaseToSuper4_MoreDerived : BaseToSuper4_DerivedA, public BaseToSuper4_DerivedB {};
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:7: ClassDecl=BaseToSuper4_MoreDerived:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="7"><Name>BaseToSuper4_Base</Name><USR>c:@S@BaseToSuper4_Base</USR><Declaration>class BaseToSuper4_MoreDerived : BaseToSuper4_DerivedA,\n                                 public BaseToSuper4_DerivedB {}</Declaration><Abstract><Para> BaseToSuper4_Base</Para></Abstract></Class>]

//===---
// Check the representation of \todo in XML.
//===---

/// Aaa.
/// \todo Bbb.
void comment_to_xml_conversion_todo_1();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_todo_1:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_todo_1</Name><USR>c:@F@comment_to_xml_conversion_todo_1#</USR><Declaration>void comment_to_xml_conversion_todo_1()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Discussion><Para kind="todo"> Bbb.</Para></Discussion></Function>]

/// Aaa.
/// \todo Bbb.
///
/// Ccc.
void comment_to_xml_conversion_todo_2();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_todo_2:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_todo_2</Name><USR>c:@F@comment_to_xml_conversion_todo_2#</USR><Declaration>void comment_to_xml_conversion_todo_2()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Discussion><Para kind="todo"> Bbb.</Para><Para> Ccc.</Para></Discussion></Function>]

/// Aaa.
/// \todo Bbb.
///
/// Ccc.
/// \todo Ddd.
void comment_to_xml_conversion_todo_3();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_todo_3:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_todo_3</Name><USR>c:@F@comment_to_xml_conversion_todo_3#</USR><Declaration>void comment_to_xml_conversion_todo_3()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Discussion><Para kind="todo"> Bbb.</Para><Para> Ccc. </Para><Para kind="todo"> Ddd.</Para></Discussion></Function>]

/// Aaa.
/// \todo Bbb.
/// \todo Ccc.
void comment_to_xml_conversion_todo_4();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_todo_4:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_todo_4</Name><USR>c:@F@comment_to_xml_conversion_todo_4#</USR><Declaration>void comment_to_xml_conversion_todo_4()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Discussion><Para kind="todo"> Bbb. </Para><Para kind="todo"> Ccc.</Para></Discussion></Function>]


//===---
// Test the representation of exception specifications in AST and XML.
//===---

/// Aaa.
/// \throws Bbb.
void comment_to_xml_conversion_exceptions_1();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_exceptions_1:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_exceptions_1</Name><USR>c:@F@comment_to_xml_conversion_exceptions_1#</USR><Declaration>void comment_to_xml_conversion_exceptions_1()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Exceptions><Para> Bbb.</Para></Exceptions></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throws]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
/// \throw Bbb.
void comment_to_xml_conversion_exceptions_2();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_exceptions_2:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_exceptions_2</Name><USR>c:@F@comment_to_xml_conversion_exceptions_2#</USR><Declaration>void comment_to_xml_conversion_exceptions_2()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Exceptions><Para> Bbb.</Para></Exceptions></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throw]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
/// \exception Bbb.
void comment_to_xml_conversion_exceptions_3();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_exceptions_3:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_exceptions_3</Name><USR>c:@F@comment_to_xml_conversion_exceptions_3#</USR><Declaration>void comment_to_xml_conversion_exceptions_3()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Exceptions><Para> Bbb.</Para></Exceptions></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[exception]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]

/// Aaa.
/// \throws Bbb.
/// \throws Ccc.
/// \throws Ddd.
void comment_to_xml_conversion_exceptions_4();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_exceptions_4:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_exceptions_4</Name><USR>c:@F@comment_to_xml_conversion_exceptions_4#</USR><Declaration>void comment_to_xml_conversion_exceptions_4()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Exceptions><Para> Bbb. </Para><Para> Ccc. </Para><Para> Ddd.</Para></Exceptions></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throws]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throws]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throws]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ddd.]))))]

/// Aaa.
/// \throws Bbb.
/// \throw Ccc.
void comment_to_xml_conversion_exceptions_5();
// CHECK: comment-to-html-xml-conversion.cpp:[[@LINE-1]]:6: FunctionDecl=comment_to_xml_conversion_exceptions_5:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}comment-to-html-xml-conversion.cpp" line="[[@LINE-1]]" column="6"><Name>comment_to_xml_conversion_exceptions_5</Name><USR>c:@F@comment_to_xml_conversion_exceptions_5#</USR><Declaration>void comment_to_xml_conversion_exceptions_5()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Exceptions><Para> Bbb. </Para><Para> Ccc.</Para></Exceptions></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throws]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[throw]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc.]))))]

#endif

