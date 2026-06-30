// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 %s -verify

typedef unsigned mode_t;

// A struct member with function type is rejected at ActOnField and becomes
// a FieldDecl, not a record-context FunctionDecl, so the call below never
// reaches the fortify gate. Regression test that this error-recovery path
// does not crash (would otherwise risk the assert in Decl.cpp isDeclExternC).
struct S {
  mode_t umask(mode_t); // expected-error {{field 'umask' declared as a function}}
};

void call_member_umask(struct S *s) {
  (void)s->umask(0xFFFF);
}
