// RUN: %clang_cc1 %s -fsyntax-only -verify -Wreturn-type -Wmissing-noreturn -Wno-unreachable-code -Wno-covered-switch-default
// RUN: %clang_cc1 %s -fsyntax-only -std=c++11 -verify -Wreturn-type -Wmissing-noreturn -Wno-unreachable-code -Wno-covered-switch-default

// A destructor may be marked noreturn and should still influence the CFG.
void pr6884_abort() __attribute__((noreturn));

struct pr6884_abort_struct {
  pr6884_abort_struct() {}
  ~pr6884_abort_struct() __attribute__((noreturn)) { pr6884_abort(); }
};

struct other { ~other() {} };

// Ensure that destructors from objects are properly modeled in the CFG despite
// the presence of switches, case statements, labels, and blocks. These tests
// try to cover bugs reported in both PR6884 and PR10063.
namespace abort_struct_complex_cfgs {
  int basic(int x) {
    switch (x) { default: pr6884_abort(); }
  }
  int f1(int x) {
    switch (x) default: pr6884_abort_struct();
  }
  int f2(int x) {
    switch (x) { default: pr6884_abort_struct(); }
  }
  int f2_positive(int x) {
    switch (x) { default: ; }
  } // expected-warning {{control reaches end of non-void function}}
  int f3(int x) {
    switch (x) { default: { pr6884_abort_struct(); } }
  }
  int f4(int x) {
    switch (x) default: L1: L2: case 4: pr6884_abort_struct();
  }
  int f5(int x) {
    switch (x) default: L1: { L2: case 4: pr6884_abort_struct(); }
  }
  int f6(int x) {
    switch (x) default: L1: L2: case 4: { pr6884_abort_struct(); }
  }

  // Test that these constructs work even when extraneous blocks are created
  // before and after the switch due to implicit destructors.
  int g1(int x) {
    other o;
    switch (x) default: pr6884_abort_struct();
  }
  int g2(int x) {
    other o;
    switch (x) { default: pr6884_abort_struct(); }
  }
  int g2_positive(int x) {
    other o;
    switch (x) { default: ; }
  } // expected-warning {{control reaches end of non-void function}}
  int g3(int x) {
    other o;
    switch (x) { default: { pr6884_abort_struct(); } }
  }
  int g4(int x) {
    other o;
    switch (x) default: L1: L2: case 4: pr6884_abort_struct();
  }
  int g5(int x) {
    other o;
    switch (x) default: L1: { L2: case 4: pr6884_abort_struct(); }
  }
  int g6(int x) {
    other o;
    switch (x) default: L1: L2: case 4: { pr6884_abort_struct(); }
  }

  // Test that these constructs work even with variables carrying the no-return
  // destructor instead of temporaries.
  int h1(int x) {
    other o;
    switch (x) default: pr6884_abort_struct a;
  }
  int h2(int x) {
    other o;
    switch (x) { default: pr6884_abort_struct a; }
  }
  int h3(int x) {
    other o;
    switch (x) { default: { pr6884_abort_struct a; } }
  }
  int h4(int x) {
    other o;
    switch (x) default: L1: L2: case 4: pr6884_abort_struct a;
  }
  int h5(int x) {
    other o;
    switch (x) default: L1: { L2: case 4: pr6884_abort_struct a; }
  }
  int h6(int x) {
    other o;
    switch (x) default: L1: L2: case 4: { pr6884_abort_struct a; }
  }
}

// PR9380
struct PR9380 {
  ~PR9380();
};
struct PR9380_B : public PR9380 {
  PR9380_B( const PR9380& str );
};
void test_PR9380(const PR9380& aKey) {
  const PR9380& flatKey = PR9380_B(aKey);
}

// Array of objects with destructors.  This is purely a coverage test case.
void test_array() {
  PR9380 a[2];
}

// Test classes wrapped in typedefs.  This is purely a coverage test case
// for CFGImplictDtor::getDestructorDecl().
void test_typedefs() {
  typedef PR9380 PR9380_Ty;
  PR9380_Ty test;
  PR9380_Ty test2[20];
}

// PR9412 - Handle CFG traversal with null successors.
enum PR9412_MatchType { PR9412_Exact };

template <PR9412_MatchType type> int PR9412_t() {
  switch (type) {
    case PR9412_Exact:
    default:
        break;
  }
} // expected-warning {{control reaches end of non-void function}}

void PR9412_f() {
    PR9412_t<PR9412_Exact>(); // expected-note {{in instantiation of function template specialization 'PR9412_t<0>' requested here}}
}

