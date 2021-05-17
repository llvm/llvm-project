// Run lines are sensitive to line numbers and come below the code.

#ifndef HEADER
#define HEADER

// Not a Doxygen comment.  notdoxy1 NOT_DOXYGEN
void notdoxy1(void);

/* Not a Doxygen comment.  notdoxy2 NOT_DOXYGEN */
void notdoxy2(void);

/*/ Not a Doxygen comment.  notdoxy3 NOT_DOXYGEN */
void notdoxy3(void);

/** Doxygen comment.  isdoxy4 IS_DOXYGEN_SINGLE */
void isdoxy4(void);

/*! Doxygen comment.  isdoxy5 IS_DOXYGEN_SINGLE */
void isdoxy5(void);

/// Doxygen comment.  isdoxy6 IS_DOXYGEN_SINGLE
void isdoxy6(void);

/* BLOCK_ORDINARY_COMMENT */
// ORDINARY COMMENT
/// This is a BCPL comment. IS_DOXYGEN_START
/// It has only two lines.
/** But there are other blocks that are part of the comment, too.  IS_DOXYGEN_END */
void multi_line_comment_plus_ordinary(int);

// MULTILINE COMMENT
//
// WITH EMPTY LINE
void multi_line_comment_empty_line(int);

int notdoxy7; // Not a Doxygen juxtaposed comment.  notdoxy7 NOT_DOXYGEN
int notdoxy8; // Not a Doxygen juxtaposed comment.  notdoxy8 NOT_DOXYGEN

int trdoxy9;  /// A Doxygen non-trailing comment.  trdoxyA IS_DOXYGEN_SINGLE
int trdoxyA;

int trdoxyB;  // Not a Doxygen trailing comment.  PART_ONE
              // It's a multiline one too.  trdoxyB NOT_DOXYGEN
int trdoxyC;

int trdoxyD;  // Not a Doxygen trailing comment.   trdoxyD NOT_DOXYGEN
              /// This comment doesn't get merged.   trdoxyE IS_DOXYGEN
int trdoxyE;

int trdoxyF;  /// A Doxygen non-trailing comment that gets dropped on the floor.
              // This comment will also be dropped.
int trdoxyG;  // This one won't.  trdoxyG NOT_DOXYGEN

int trdoxyH;  ///< A Doxygen trailing comment.  PART_ONE
              // This one gets merged with it.  trdoxyH SOME_DOXYGEN
int trdoxyI;  // This one doesn't.  trdoxyI NOT_DOXYGEN

int trdoxyJ;  // Not a Doxygen trailing comment.  PART_ONE
              ///< This one gets merged with it.  trdoxyJ SOME_DOXYGEN
int trdoxyK;  // This one doesn't.  trdoxyK NOT_DOXYGEN

int trdoxyL;  // Not a Doxygen trailing comment.  trdoxyL NOT_DOXYGEN
// This one shouldn't get merged.  trdoxyM NOT_DOXYGEN
int trdoxyM;

int trdoxyN;  ///< A Doxygen trailing comment.  trdoxyN IS_DOXYGEN
  // This one shouldn't get merged.  trdoxyO NOT_DOXYGEN
int trdoxyO;


#endif

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: c-index-test -write-pch %t/out.pch -fparse-all-comments -x c++ -std=c++11 %s

// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -x c++ -std=c++11 %s -fparse-all-comments > %t/out.c-index-direct
// RUN: c-index-test -test-load-tu %t/out.pch all > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-pch

// CHECK: parse-all-comments.c:7:6: FunctionDecl=notdoxy1:{{.*}} notdoxy1 NOT_DOXYGEN
// CHECK: parse-all-comments.c:10:6: FunctionDecl=notdoxy2:{{.*}} notdoxy2 NOT_DOXYGEN
// CHECK: parse-all-comments.c:13:6: FunctionDecl=notdoxy3:{{.*}} notdoxy3 NOT_DOXYGEN
// CHECK: parse-all-comments.c:16:6: FunctionDecl=isdoxy4:{{.*}} isdoxy4 IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:19:6: FunctionDecl=isdoxy5:{{.*}} isdoxy5 IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:22:6: FunctionDecl=isdoxy6:{{.*}} isdoxy6 IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:29:6: FunctionDecl=multi_line_comment_plus_ordinary:{{.*}} BLOCK_ORDINARY_COMMENT {{.*}} ORDINARY COMMENT {{.*}} IS_DOXYGEN_START {{.*}} IS_DOXYGEN_END
// CHECK: parse-all-comments.c:34:6: FunctionDecl=multi_line_comment_empty_line:{{.*}} MULTILINE COMMENT{{.*}}\n{{.*}}\n{{.*}} WITH EMPTY LINE
// CHECK: parse-all-comments.c:36:5: VarDecl=notdoxy7:{{.*}} notdoxy7 NOT_DOXYGEN
// CHECK: parse-all-comments.c:37:5: VarDecl=notdoxy8:{{.*}} notdoxy8 NOT_DOXYGEN
// CHECK-NOT: parse-all-comments.c:39:5: VarDecl=trdoxy9:{{.*}} trdoxyA IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:40:5: VarDecl=trdoxyA:{{.*}} trdoxyA IS_DOXYGEN_SINGLE
// CHECK: parse-all-comments.c:42:5: VarDecl=trdoxyB:{{.*}} PART_ONE {{.*}} trdoxyB NOT_DOXYGEN
// CHECK-NOT: parse-all-comments.c:44:5: VarDecl=trdoxyC:{{.*}} trdoxyB NOT_DOXYGEN
// CHECK: parse-all-comments.c:46:5: VarDecl=trdoxyD:{{.*}} trdoxyD NOT_DOXYGEN
// CHECK: parse-all-comments.c:48:5: VarDecl=trdoxyE:{{.*}} trdoxyE IS_DOXYGEN
// CHECK-NOT: parse-all-comments.c:50:5: VarDecl=trdoxyF:{{.*}} RawComment
// CHECK: parse-all-comments.c:52:5: VarDecl=trdoxyG:{{.*}} trdoxyG NOT_DOXYGEN
// CHECK: parse-all-comments.c:54:5: VarDecl=trdoxyH:{{.*}} PART_ONE {{.*}} trdoxyH SOME_DOXYGEN
// CHECK: parse-all-comments.c:56:5: VarDecl=trdoxyI:{{.*}} trdoxyI NOT_DOXYGEN
// CHECK: parse-all-comments.c:58:5: VarDecl=trdoxyJ:{{.*}} PART_ONE {{.*}} trdoxyJ SOME_DOXYGEN
// CHECK: parse-all-comments.c:60:5: VarDecl=trdoxyK:{{.*}} trdoxyK NOT_DOXYGEN
// CHECK: parse-all-comments.c:62:5: VarDecl=trdoxyL:{{.*}} trdoxyL NOT_DOXYGEN
// CHECK: parse-all-comments.c:64:5: VarDecl=trdoxyM:{{.*}} trdoxyM NOT_DOXYGEN
// CHECK: parse-all-comments.c:66:5: VarDecl=trdoxyN:{{.*}} trdoxyN IS_DOXYGEN
// CHECK: parse-all-comments.c:68:5: VarDecl=trdoxyO:{{.*}} trdoxyO NOT_DOXYGEN
