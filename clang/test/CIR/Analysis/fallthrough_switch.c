// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -fclangir-analysis="fallthrough" -emit-cir -o /dev/null -verify
// INFO: Switch statement test cases for CIR fallthrough analysis
// Derived from clang/test/Sema/return.c

// ===========================================================================
// Test cases where switch SHOULD NOT warn (all paths return)
// ===========================================================================

// Switch with default - all cases return
int switch_all_return(int x) {
  switch (x) {
  case 0:
    return 0;
  case 1:
    return 1;
  default:
    return -1;
  }
}

// Test from return.c: enum switch covering all cases
enum Cases { C1, C2, C3, C4 };
int test_enum_cases(enum Cases C) {
  switch (C) {
  case C1: return 1;
  case C2: return 2;
  case C4: return 3;
  case C3: return 4;
  }
} 
// TODO: Should not warn - enum covers all cases (needs enum coverage analysis)

// ===========================================================================
// Test cases where switch SHOULD warn
// ===========================================================================

// Switch without default
int switch_no_default(int x) {
  switch (x) {
  case 0:
    return 0;
  case 2:
    return 2;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// Switch with only default that doesn't return
int switch_default_no_return(int x) {
  switch (x) default: ;
} // expected-warning {{non-void function does not return a value}}

// Switch with break in one case
int switch_with_break(int x) {
  switch (x) {
  case 0:
    return 0;
  case 1:
    break;
  default:
    return -1;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// Fallthrough case (using fallthrough, not return)
int switch_fallthrough(int x) {
  switch (x) {
  case 0:
  case 1:  // fallthrough from case 0
    return 1;
  default:
    break;  // falls through to end
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// ===========================================================================
// Nested switch statements
// ===========================================================================

int nested_switch(int x, int y) {
  switch (x) {
  case 0:
    switch (y) {
    case 0:
      return 0;
    default:
      return 1;
    }
  default:
    return -1;
  }
}

int nested_switch_missing(int x, int y) {
  switch (x) {
  case 0:
    switch (y) {
    case 0:
      return 0;
    // No default in inner switch
    }
    break;  // Falls through after inner switch
  default:
    return -1;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

