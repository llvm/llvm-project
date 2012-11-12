// RUN: c-index-test -test-load-source all %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -check-prefix=ERR %s

// CHECK: annotate-comments-unterminated.c:9:5: VarDecl=x:{{.*}} RawComment=[/** Aaa. */]{{.*}} BriefComment=[Aaa.]
// CHECK: annotate-comments-unterminated.c:11:5: VarDecl=y:{{.*}} RawComment=[/**< Bbb. */]{{.*}} BriefComment=[Bbb.]
// CHECK-ERR: error: unterminated

/** Aaa. */
int x;

int y; /**< Bbb. */
/**< Ccc.
 * Ddd.
