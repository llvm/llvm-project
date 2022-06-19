// RUN: %clang_cc1 -std=c++2b -verify %s

void test_consteval() {
  if consteval ({(void)1;});  // expected-error {{expected { after consteval}}
  if consteval (void) 0; // expected-error {{expected { after consteval}}
  if consteval {
    (void)0;
  } else (void)0; // expected-error {{expected { after else}}

  static_assert([] {
    if consteval {
      return 0;
    }
    return 1;
  }() == 0);

  static_assert([] {
    if consteval {
      return 0;
    } else {
      return 1;
    }
  }() == 0);

  static_assert([] {
    if !consteval {
      return 0;
    } else {
      return 1;
    }
  }() == 1);

  static_assert([] {
    if not consteval {
      return 0;
    }
    return 1;
  }() == 1);

  if consteval [[likely]] { // expected-warning {{attribute 'likely' has no effect when annotating an 'if consteval' statement}}\
                            // expected-note 2{{annotating the 'if consteval' statement here}}


  }
  else [[unlikely]] { // expected-warning {{attribute 'unlikely' has no effect when annotating an 'if consteval' statement}}

  }

}

void test_consteval_jumps() {
  if consteval { // expected-note 4{{jump enters controlled statement of consteval if}}
    goto a;
    goto b; // expected-error {{cannot jump from this goto statement to its label}}
  a:;
  } else {
    goto b;
    goto a; // expected-error {{cannot jump from this goto statement to its label}}
  b:;
  }
  goto a; // expected-error {{cannot jump from this goto statement to its label}}
  goto b; // expected-error {{cannot jump from this goto statement to its label}}
}

void test_consteval_switch() {
  int x = 42;
  switch (x) {
    if consteval { // expected-note 2{{jump enters controlled statement of consteval if}}
    case 1:;       // expected-error {{cannot jump from switch statement to this case label}}
    default:;      // expected-error {{cannot jump from switch statement to this case label}}
    } else {
    }
  }
  switch (x) {
    if consteval { // expected-note 2{{jump enters controlled statement of consteval if}}
    } else {
    case 2:;  // expected-error {{cannot jump from switch statement to this case label}}
    default:; // expected-error {{cannot jump from switch statement to this case label}}
    }
  }
}

consteval int f(int i) { return i; }
constexpr int g(int i) {
  if consteval {
    return f(i);
  } else {
    return 42;
  }
}
static_assert(g(10) == 10);

constexpr int h(int i) { // expected-note {{declared here}}
  if !consteval {
    return f(i); // expected-error {{call to consteval function 'f' is not a constant expression}}\
                     // expected-note  {{cannot be used in a constant expression}}
  }
  return 0;
}

consteval void warn_in_consteval() {
  if consteval { // expected-warning {{consteval if is always true in an immediate context}}
    if consteval {} // expected-warning {{consteval if is always true in an immediate context}}
  }
}

constexpr void warn_in_consteval2() {
  if consteval {
    if consteval {} // expected-warning {{consteval if is always true in an immediate context}}
  }
}

auto y = []() consteval {
  if consteval { // expected-warning {{consteval if is always true in an immediate context}}
    if consteval {} // expected-warning {{consteval if is always true in an immediate context}}
  }
};

namespace test_transform {
int f(auto n) {
  if consteval {
    n.foo; //expected-error {{no member named}}
  }
  else {
  }

  if !consteval {
    n.foo; //expected-error {{no member named}}
  }
  else {
  }

  return 0;
}

constexpr int g(auto n) {
  if consteval {
  }
  else {
    n.foo; //expected-error {{no member named}}
  }

  if !consteval {
  }
  else {
     n.foo; //expected-error {{no member named}}
  }

  return 0;
}

struct S {};
void test() {
  f(S{}); //expected-note {{in instantiation}}
  g(S{}); //expected-note {{in instantiation}}
}

}
