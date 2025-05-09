// RUN: %clang_cc1 -fsyntax-only -verify=c,both -Wjump-misses-init %s
// RUN: %clang_cc1 -fsyntax-only -verify=c,both -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -fms-compatibility %s
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

// Statement expressions are a bit strange in that they seem to allow for
// jumping past initialization without being diagnosed, even in C++. Perhaps
// this should change?
void f(void) {
  ({
    goto ouch;
    int i = 12;
  });

  for (int i = ({ goto ouch; int x = 10; x;}); i < 0; ++i) {
  }

ouch:
  ;
}

void indirect(int n) {
DirectJump:
  ;

  void *Table[] = {&&DirectJump, &&Later};
  goto *Table[n]; // c-warning {{jump from this indirect goto statement to one of its possible targets is incompatible with C++}} \
                     cxx-error {{cannot jump from this indirect goto statement to one of its possible targets}}

  int x = 12;     // both-note {{jump bypasses variable initialization}}
Later:            // both-note {{possible target of indirect goto statement}}
  ;
}
