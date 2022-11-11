// RUN: %check_clang_tidy %s readability-isolate-declaration %t
// XFAIL: *

struct S {
  int a;
  const int b;
  void f() {}
};

void member_pointers() {
  // FIXME: Memberpointers are transformed incorrect. Emit only a warning
  // for now.
  int S::*p = &S::a, S::*const q = &S::a;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int S::*p = &S::a;
  // CHECK-FIXES: {{^  }}int S::*const q = &S::a;

  int /* :: */ S::*pp2 = &S::a, var1 = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int /* :: */ S::*pp2 = &S::a;
  // CHECK-FIXES: {{^  }}int var1 = 0;

  const int S::*r = &S::b, S::*t;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: const int S::*r = &S::b;
  // CHECK-FIXES: {{^  }}const int S::*t;

  {
    int S::*mdpa1[2] = {&S::a, &S::a}, var1 = 0;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: int S::*mdpa1[2] = {&S::a, &S::a};
    // CHECK-FIXES: {{^    }}int var1 = 0;

    int S ::**mdpa2[2] = {&p, &pp2}, var2 = 0;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: int S ::**mdpa2[2] = {&p, &pp2};
    // CHECK-FIXES: {{^    }}int var2 = 0;

    void (S::*mdfp1)() = &S::f, (S::*mdfp2)() = &S::f;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: void (S::*mdfp1)() = &S::f;
    // CHECK-FIXES: {{^    }}void (S::*mdfp2)() = &S::f;

    void (S::*mdfpa1[2])() = {&S::f, &S::f}, (S::*mdfpa2)() = &S::f;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: void (S::*mdfpa1[2])() = {&S::f, &S::f};
    // CHECK-FIXES: {{^    }}void (S::*mdfpa2)() = &S::f;

    void (S::* * mdfpa3[2])() = {&mdfpa1[0], &mdfpa1[1]}, (S::*mdfpa4)() = &S::f;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: multiple declarations in a single statement reduces readability
    // CHECK-FIXES: void (S::* * mdfpa3[2])() = {&mdfpa1[0], &mdfpa1[1]};
    // CHECK-FIXES: {{^    }}void (S::*mdfpa4)() = &S::f;
  }

  class CS {
  public:
    int a;
    const int b;
  };
  int const CS ::*pp = &CS::a, CS::*const qq = &CS::a;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-FIXES: int const CS ::*pp = &CS::a;
  // CHECK-FIXES: {{^  }}int const CS::*const qq = &CS::a;
}
