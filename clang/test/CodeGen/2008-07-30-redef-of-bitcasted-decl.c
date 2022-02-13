// RUN: %clang_cc1 -emit-llvm -o - %s
// <rdar://problem/6108358>

/* For posterity, the issue here begins initial "char []" decl for
 * s. This is a tentative definition and so a global was being
 * emitted, however the mapping in GlobalDeclMap referred to a bitcast
 * of this global.
 *
 * The problem was that later when the correct definition for s is
 * emitted we were doing a RAUW on the old global which was destroying
 * the bitcast in the GlobalDeclMap (since it cannot be replaced
 * properly), leaving a dangling pointer.
 *
 * The purpose of bar is just to trigger a use of the old decl
 * sometime after the dangling pointer has been introduced.
 */

char s[];

static void bar(void *db) {
  eek(s);
}

char s[5] = "hi";

int foo(void) {
  bar(0);
}
