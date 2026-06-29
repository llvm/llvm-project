// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 -Wno-uninitialized %s
// The ERROR run adds a leading unrelated error so every later function is
// analyzed through the post-error path; the same constructor-body diagnostics
// must still fire there.
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -Wno-uninitialized -DLEADING_ERROR %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

#ifdef LEADING_ERROR
int leading_unrelated_error = undeclared_identifier;
// expected-error@-1 {{use of undeclared identifier 'undeclared_identifier'}}
#endif

namespace std { enum class byte : unsigned char {}; }

// A [[uninit]] member that is never read needs no assignment: the constructor
// is not required to initialize it (paper §5.1/§5.3).
struct NeverReadEmpty {
  int m [[uninit]];
  NeverReadEmpty() {}
};

struct NeverReadActive {
  int m [[uninit]];
  int other = 0;
  NeverReadActive() { other = 1; }
};

struct ReadAfterAssign {
  int m [[uninit]];
  ReadAfterAssign() { m = 1; int y = m; (void)y; }
};

struct ReadBeforeAssign {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  ReadBeforeAssign() { int y = m; (void)y; m = 1; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct SelfReadOnRHS {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  SelfReadOnRHS() { m = m + 1; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct CompoundAssignReads {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  CompoundAssignReads() { m += 1; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct OneBranchThenRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  OneBranchThenRead(bool b) {
    if (b)
      m = 1;
    int y = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)y;
  }
};

struct BothBranchesThenRead {
  int m [[uninit]];
  BothBranchesThenRead(bool b) {
    if (b)
      m = 1;
    else
      m = 2;
    int y = m;
    (void)y;
  }
};

struct LoopBodyThenRead {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  LoopBodyThenRead(int n) {
    for (int i = 0; i < n; ++i)
      m = i;
    int y = m; // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
    (void)y;
  }
};

// std::byte may be read while uninitialized (paper §4.5).
struct ByteExempt {
  std::byte b [[uninit]];
  ByteExempt() { std::byte c = b; (void)c; }
};

// A member initialized in the (written) member-initializer list is assigned at
// body entry, so a later read is fine and no marker/list-init contradiction is
// introduced.
struct MarkerWithListInit {
  int m [[uninit]];
  MarkerWithListInit() : m(0) { int y = m; (void)y; }
};

struct MultipleMembers {
  int a [[uninit]];
  int b [[uninit]]; // expected-note {{member 'b' declared here}}
  int c [[uninit]];
  MultipleMembers() {
    a = 1;
    int x = a; (void)x;
    int y = b; (void)y; // expected-error {{member 'b' is read before initialization under profile 'std::init'}}
    c = 2;
    int z = c; (void)z;
  }
};

struct ExplicitThis {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  ExplicitThis() { int y = this->m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};

struct OutOfLine {
  int m [[uninit]]; // expected-note {{member 'm' declared here}}
  OutOfLine();
};
OutOfLine::OutOfLine() { int y = m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}

template <typename T>
struct Tmpl {
  T m [[uninit]]; // expected-note {{member 'm' declared here}}
  Tmpl() { T y = m; (void)y; } // expected-error {{member 'm' is read before initialization under profile 'std::init'}}
};
template struct Tmpl<int>; // expected-note {{in instantiation of member function 'Tmpl<int>::Tmpl' requested here}}

struct SuppressedCtor {
  int m [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] SuppressedCtor() { int y = m; (void)y; }
};

struct SuppressedByRule {
  int m [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_read")]] SuppressedByRule() { int y = m; (void)y; }
};

struct SuppressedStmt {
  int m [[uninit]];
  SuppressedStmt() {
    // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
    [[profiles::suppress(std::init)]] { int y = m; (void)y; }
  }
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] SuppressedClass {
  int m [[uninit]];
  SuppressedClass() { int y = m; (void)y; }
};
