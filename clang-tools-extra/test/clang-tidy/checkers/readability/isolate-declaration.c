// RUN: %check_clang_tidy %s readability-isolate-declaration %t

void c_specific(void) {
  void (*signal(int sig, void (*func)(int)))(int);
  int i = sizeof(struct S { int i; });
  int j = sizeof(struct T { int i; }), k;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int j = sizeof(struct T { int i; });
  // CHECK-FIXES: {{^  }}int k;

  void g(struct U { int i; } s);                // One decl
  void h(struct V { int i; } s), m(int i, ...); // Two decls
}
