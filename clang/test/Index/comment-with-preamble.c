// Make sure the preable does not truncate comments.

#ifndef BAZ
#define BAZ 3
#endif

//! Foo’s description.
void Foo();

// RUN: c-index-test -test-load-source-reparse 1 local %s | FileCheck %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 1 local %s | FileCheck %s

// CHECK: FunctionDecl=Foo:8:6 RawComment=[//! Foo’s description.] RawCommentRange=[7:1 - 7:25] BriefComment=[Foo’s description.]
