// Comments are only collected into the AST when a consumer may read them back
// (see Sema::shouldRetainCommentsInAST). -ast-dump is such a consumer: it
// force-enables comment retention, so documentation comments remain visible in
// its output even without -Wdocumentation. An ordinary comment is only turned
// into an AST comment node when -fparse-all-comments is passed.

// RUN: %clang_cc1 -ast-dump -ast-dump-filter Test %s \
// RUN:   | FileCheck -strict-whitespace %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang_cc1 -fparse-all-comments -ast-dump -ast-dump-filter Test %s \
// RUN:   | FileCheck -strict-whitespace %s --check-prefixes=CHECK,ALL

/// Doc
int Test_DocComment;
// A documentation comment is retained for -ast-dump in both modes.
// CHECK:      VarDecl{{.*}}Test_DocComment
// CHECK-NEXT:   FullComment
// CHECK-NEXT:     ParagraphComment
// CHECK-NEXT:       TextComment{{.*}} Text=" Doc"

// Ordinary
int Test_OrdinaryComment;
// An ordinary comment becomes an AST comment node only with
// -fparse-all-comments; by default it is dropped.
// CHECK:       VarDecl{{.*}}Test_OrdinaryComment
// ALL-NEXT:      FullComment
// ALL-NEXT:        ParagraphComment
// ALL-NEXT:          TextComment{{.*}} Text=" Ordinary"
// DEFAULT-NOT: FullComment
