// RUN: %clang_cc1 -fsyntax-only -verify=c,both -Wjump-bypasses-init %s
// RUN: %clang_cc1 -fsyntax-only -verify=c,both -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx,both -x c++ %s
// good-no-diagnostics

void goto_func_1(void) {
  goto ouch;  // c-warning {{jump from this goto statement to its label is incompatible with C++}} \
                 cxx-error {{cannot jump from this goto statement to its label}}
  int i = 12; // both-note {{jump bypasses variable initialization}}

ouch:
  ;
}

void goto_func_2(void) {
  goto ouch;
  static int i = 12; // This initialization is not jumped over, so no warning.

ouch:
  ;
}

void switch_func(int i) {
  switch (i) {
    int x = 12; // both-note {{jump bypasses variable initialization}}
  case 0:       // c-warning {{jump from switch statement to this case label is incompatible with C++}} \
                   cxx-error {{cannot jump from switch statement to this case label}}
    break;
  }
}
