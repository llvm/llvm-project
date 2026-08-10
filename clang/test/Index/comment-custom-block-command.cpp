// RUN: rm -rf %t
// RUN: mkdir %t

// Check that custom block commands are defined correctly.
// RUN: %clang_cc1 -fcomment-block-commands=CustomCommand -x c++ -std=c++11 -emit-pch -o %t/out.pch %s
// RUN: %clang_cc1 -x c++ -std=c++11 -fcomment-block-commands=CustomCommand -include-pch %t/out.pch -fsyntax-only %s

// RUN: c-index-test -write-pch %t/out.pch -fcomment-block-commands=CustomCommand -x c++ -std=c++11 %s
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -std=c++11 -fcomment-block-commands=CustomCommand > %t/out.c-index-direct
// RUN: c-index-test -test-load-tu %t/out.pch all > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-pch

// XFAIL: vg_leak

#ifndef HEADER
#define HEADER

/// \CustomCommand Aaa.
void comment_custom_block_command_1();

// CHECK: comment-custom-block-command.cpp:[[@LINE-2]]:6: FunctionDecl=comment_custom_block_command_1:{{.*}} FullCommentAsHTML=[<p> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}comment-custom-block-command.cpp" line="[[@LINE-2]]" column="6"><Name>comment_custom_block_command_1</Name><USR>c:@F@comment_custom_block_command_1#</USR><Declaration>void comment_custom_block_command_1()</Declaration><Discussion><Para> Aaa.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[CustomCommand]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]

#endif

