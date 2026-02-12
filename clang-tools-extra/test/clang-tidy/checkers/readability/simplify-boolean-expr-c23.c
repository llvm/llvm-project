// RUN: %check_clang_tidy -std=c23-or-later %s readability-simplify-boolean-expr %t

bool a1 = false;

void if_with_bool_literal_condition() {
  int i = 0;
  if (false) {
    i = 1;
  } else {
    i = 2;
  }
  i = 3;
  // CHECK-MESSAGES: :[[@LINE-6]]:7: warning: redundant boolean literal in if statement condition
  // CHECK-FIXES:      {{^  int i = 0;$}}
  // CHECK-FIXES-NEXT: {{^  {$}}
  // CHECK-FIXES-NEXT: {{^    i = 2;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
  // CHECK-FIXES-NEXT: {{^  i = 3;$}}

  i = 4;
  if (true) {
    i = 5;
  } else {
    i = 6;
  }
  i = 7;
  // CHECK-MESSAGES: :[[@LINE-6]]:7: warning: redundant boolean literal in if statement condition
  // CHECK-FIXES:      {{^  i = 4;$}}
  // CHECK-FIXES-NEXT: {{^  {$}}
  // CHECK-FIXES-NEXT: {{^    i = 5;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
  // CHECK-FIXES-NEXT: {{^  i = 7;$}}

  i = 8;
  if (false) {
    i = 9;
  }
  i = 11;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant boolean literal in if statement condition
  // CHECK-FIXES:      {{^  i = 8;$}}
  // CHECK-FIXES-NEXT: {{^  $}}
  // CHECK-FIXES-NEXT: {{^  i = 11;$}}
}

void if_with_negated_bool_condition() {
  int i = 10;
  if (!true) {
    i = 11;
  } else {
    i = 12;
  }
  i = 13;
  // CHECK-MESSAGES: :[[@LINE-6]]:7: warning: redundant boolean literal in if statement condition
  // CHECK-FIXES:      {{^  int i = 10;$}}
  // CHECK-FIXES-NEXT: {{^  {$}}
  // CHECK-FIXES-NEXT: {{^    i = 12;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
  // CHECK-FIXES-NEXT: {{^  i = 13;$}}

  i = 14;
  if (!false) {
    i = 15;
  } else {
    i = 16;
  }
  i = 17;
  // CHECK-MESSAGES: :[[@LINE-6]]:7: warning: redundant boolean literal in if statement condition
  // CHECK-FIXES:      {{^  i = 14;$}}
  // CHECK-FIXES-NEXT: {{^  {$}}
  // CHECK-FIXES-NEXT: {{^    i = 15;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
  // CHECK-FIXES-NEXT: {{^  i = 17;$}}

  i = 18;
  if (!true) {
    i = 19;
  }
  i = 20;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant boolean literal in if statement condition
  // CHECK-FIXES:      {{^  i = 18;$}}
  // CHECK-FIXES-NEXT: {{^  $}}
  // CHECK-FIXES-NEXT: {{^  i = 20;$}}
}

void operator_equals() {
  int i = 0;
  bool b1 = (i > 2);
  if (b1 == true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(b1\) {$}}
    i = 5;
  } else {
    i = 6;
  }
  bool b2 = (i > 4);
  if (b2 == false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(!b2\) {$}}
    i = 7;
  } else {
    i = 9;
  }
  bool b3 = (i > 6);
  if (true == b3) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(b3\) {$}}
    i = 10;
  } else {
    i = 11;
  }
  bool b4 = (i > 8);
  if (false == b4) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(!b4\) {$}}
    i = 12;
  } else {
    i = 13;
  }
}

void operator_or() {
  int i = 0;
  bool b5 = (i > 10);
  if (b5 || false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(b5\) {$}}
    i = 14;
  } else {
    i = 15;
  }
  bool b6 = (i > 10);
  if (b6 || true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(true\) {$}}
    i = 16;
  } else {
    i = 17;
  }
  bool b7 = (i > 10);
  if (false || b7) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(b7\) {$}}
    i = 18;
  } else {
    i = 19;
  }
  bool b8 = (i > 10);
  if (true || b8) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(true\) {$}}
    i = 20;
  } else {
    i = 21;
  }
}

void operator_and() {
  int i = 0;
  bool b9 = (i > 20);
  if (b9 && false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(false\) {$}}
    i = 22;
  } else {
    i = 23;
  }
  bool ba = (i > 20);
  if (ba && true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(ba\) {$}}
    i = 24;
  } else {
    i = 25;
  }
  bool bb = (i > 20);
  if (false && bb) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(false\) {$}}
    i = 26;
  } else {
    i = 27;
  }
  bool bc = (i > 20);
  if (true && bc) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(bc\) {$}}
    i = 28;
  } else {
    i = 29;
  }
}

void ternary_operator() {
  int i = 0;
  bool bd = (i > 20) ? true : false;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: redundant boolean literal in ternary expression result
  // CHECK-FIXES: {{^  bool bd = i > 20;$}}

  bool be = (i > 20) ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: redundant boolean literal in ternary expression result
  // CHECK-FIXES: {{^  bool be = i <= 20;$}}

  bool bf = ((i > 20)) ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: redundant boolean literal in ternary expression result
  // CHECK-FIXES: {{^  bool bf = i <= 20;$}}
}

void operator_not_equal() {
  int i = 0;
  bool bf = (i > 20);
  if (false != bf) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(bf\) {$}}
    i = 30;
  } else {
    i = 31;
  }
  bool bg = (i > 20);
  if (true != bg) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(!bg\) {$}}
    i = 32;
  } else {
    i = 33;
  }
  bool bh = (i > 20);
  if (bh != false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(bh\) {$}}
    i = 34;
  } else {
    i = 35;
  }
  bool bi = (i > 20);
  if (bi != true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(!bi\) {$}}
    i = 36;
  } else {
    i = 37;
  }
}

void nested_booleans() {
  if (false || (true || false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(false \|\| \(true\)\) {$}}
  }
  if (true && (true || false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(true && \(true\)\) {$}}
  }
  if (false || (true && false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(false \|\| \(false\)\) {$}}
  }
  if (true && (true && false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant boolean literal supplied to boolean operator
    // CHECK-FIXES: {{^  if \(true && \(false\)\) {$}}
  }
}

#define HAS_XYZ_FEATURE true
#define M1(what) M2(true, what)
#define M2(condition, what) if (condition) what

void macros() {
  int i = 0;
  bool b = (i == 1);
  i = 2;
  if (b && HAS_XYZ_FEATURE) {
    // leave this alone; if you want it simplified, then you should
    // inline the macro first.
    i = 3;
  }
  if (HAS_XYZ_FEATURE) {
    i = 5;
  }
  i = 4;
  M1(i = 7);
}

#undef HAS_XYZ_FEATURE

bool conditional_return_statements(int i) {
  if (i == 0) return true; else return false;
}
// CHECK-MESSAGES: :[[@LINE-2]]:22: warning: redundant boolean literal in conditional return statement
// CHECK-FIXES:      {{^  return i == 0;$}}

bool conditional_return_statements_no_fix_1(int i) {
  if (i == 0) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant boolean literal in conditional return statement
  // CHECK-MESSAGES: :[[@LINE-2]]:7: note: conditions that can be simplified
  // comment
  return false;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: return statement that can be simplified
}

bool conditional_return_statements_no_fix_2(int i) {
  if (i == 0) return true;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant boolean literal in conditional return statement
  // CHECK-MESSAGES: :[[@LINE-2]]:7: note: conditions that can be simplified
  // comment
  else return false;
}

bool conditional_return_statements_then_expr(int i, int j) {
  if (i == j) return (i == 0); else return false;
}

bool conditional_return_statements_else_expr(int i, int j) {
  if (i == j) return true; else return (i == 0);
}

bool negated_conditional_return_statements(int i) {
  if (i == 0) return false; else return true;
}
// CHECK-MESSAGES: :[[@LINE-2]]:22: warning: redundant boolean literal in conditional return statement
// CHECK-FIXES:      {{^  return i != 0;$}}

bool negative_condition_conditional_return_statement(int i) {
  if (!(i == 0)) return false; else return true;
}
// CHECK-MESSAGES: :[[@LINE-2]]:25: warning: redundant boolean literal in conditional return statement
// CHECK-FIXES:      {{^  return i == 0;$}}

bool conditional_compound_return_statements(int i) {
  if (i == 1) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return statement
// CHECK-FIXES:      {{^  return i == 1;$}}

bool negated_conditional_compound_return_statements(int i) {
  if (i == 1) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return statement
// CHECK-FIXES:      {{^  return i != 1;$}}

bool conditional_return_statements_side_effects_then(int i) {
  if (i == 2) {
    macros();
    return true;
  } else
    return false;
}

bool negated_conditional_return_statements_side_effects_then(int i) {
  if (i == 2) {
    macros();
    return false;
  } else
    return true;
}

bool conditional_return_statements_side_effects_else(int i) {
  if (i == 2)
    return true;
  else {
    macros();
    return false;
  }
}

bool negated_conditional_return_statements_side_effects_else(int i) {
  if (i == 2)
    return false;
  else {
    macros();
    return true;
  }
}

void simple_conditional_assignment_statements(int i) {
  bool b;
  if (i > 10)
    b = true;
  else
    b = false;
  bool bb = false;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: redundant boolean literal in conditional assignment
  // CHECK-FIXES: bool b;
  // CHECK-FIXES: {{^  b = i > 10;$}}
  // CHECK-FIXES: bool bb = false;

  bool c;
  if (i > 20)
    c = false;
  else
    c = true;
  bool c2 = false;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: redundant boolean literal in conditional assignment
  // CHECK-FIXES: bool c;
  // CHECK-FIXES: {{^  c = i <= 20;$}}
  // CHECK-FIXES: bool c2 = false;

  // Unchanged: different variables.
  bool b2;
  if (i > 12)
    b = true;
  else
    b2 = false;

  // Unchanged: no else statement.
  bool b3;
  if (i > 15)
    b3 = true;

  // Unchanged: not boolean assignment.
  int j;
  if (i > 17)
    j = 10;
  else
    j = 20;

  // Unchanged: different variables assigned.
  int k = 0;
  bool b4 = false;
  if (i > 10)
    b4 = true;
  else
    k = 10;
}

void complex_conditional_assignment_statements(int i) {
  bool d;
  if (i > 30) {
    d = true;
  } else {
    d = false;
  }
  d = false;
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: redundant boolean literal in conditional assignment
  // CHECK-FIXES: bool d;
  // CHECK-FIXES: {{^  d = i > 30;$}}
  // CHECK-FIXES: d = false;

  bool e;
  if (i > 40) {
    e = false;
  } else {
    e = true;
  }
  e = false;
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: redundant boolean literal in conditional assignment
  // CHECK-FIXES: bool e;
  // CHECK-FIXES: {{^  e = i <= 40;$}}
  // CHECK-FIXES: e = false;

  // Unchanged: no else statement.
  bool b3;
  if (i > 15) {
    b3 = true;
  }

  // Unchanged: not a boolean assignment.
  int j;
  if (i > 17) {
    j = 10;
  } else {
    j = 20;
  }

  // Unchanged: multiple statements.
  bool f;
  if (j > 10) {
    j = 10;
    f = true;
  } else {
    j = 20;
    f = false;
  }

  // Unchanged: multiple statements.
  bool g;
  if (j > 10)
    g = true;
  else {
    j = 20;
    g = false;
  }

  // Unchanged: multiple statements.
  bool h;
  if (j > 10) {
    j = 10;
    h = true;
  } else
    h = false;
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
bool chained_conditional_compound_return(int i) {
  if (i < 0) {
    return true;
  } else if (i < 10) {
    return false;
  } else if (i > 20) {
    return true;
  } else {
    return false;
  }
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
bool chained_conditional_return(int i) {
  if (i < 0)
    return true;
  else if (i < 10)
    return false;
  else if (i > 20)
    return true;
  else
    return false;
}

// Unchanged: chained assignments, but ChainedConditionalAssignment not set.
void chained_conditional_compound_assignment(int i) {
  bool b;
  if (i < 0) {
    b = true;
  } else if (i < 10) {
    b = false;
  } else if (i > 20) {
    b = true;
  } else {
    b = false;
  }
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
void chained_conditional_assignment(int i) {
  bool b;
  if (i < 0)
    b = true;
  else if (i < 10)
    b = false;
  else if (i > 20)
    b = true;
  else
    b = false;
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
bool chained_simple_if_return_negated(int i) {
  if (i < 5)
    return false;
  if (i > 10)
    return false;
  return true;
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
bool complex_chained_if_return_return(int i) {
  if (i < 5) {
    return true;
  }
  if (i > 10) {
    return true;
  }
  return false;
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
bool complex_chained_if_return_return_negated(int i) {
  if (i < 5) {
    return false;
  }
  if (i > 10) {
    return false;
  }
  return true;
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
bool chained_simple_if_return(int i) {
  if (i < 5)
    return true;
  if (i > 10)
    return true;
  return false;
}

bool simple_if_return_return(int i) {
  if (i > 10)
    return true;
  return false;
}
// CHECK-MESSAGES: :[[@LINE-3]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}bool simple_if_return_return(int i) {{{$}}
// CHECK-FIXES: {{^  return i > 10;$}}
// CHECK-FIXES: {{^}$}}

bool simple_if_return_return_negated(int i) {
  if (i > 10)
    return false;
  return true;
}
// CHECK-MESSAGES: :[[@LINE-3]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}bool simple_if_return_return_negated(int i) {{{$}}
// CHECK-FIXES: {{^  return i <= 10;$}}
// CHECK-FIXES: {{^}$}}

bool complex_if_return_return(int i) {
  if (i > 10) {
    return true;
  }
  return false;
}
// CHECK-MESSAGES: :[[@LINE-4]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}bool complex_if_return_return(int i) {{{$}}
// CHECK-FIXES: {{^  return i > 10;$}}
// CHECK-FIXES: {{^}$}}

bool complex_if_return_return_negated(int i) {
  if (i > 10) {
    return false;
  }
  return true;
}
// CHECK-MESSAGES: :[[@LINE-4]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}bool complex_if_return_return_negated(int i) {{{$}}
// CHECK-FIXES: {{^  return i <= 10;$}}
// CHECK-FIXES: {{^}$}}

bool if_implicit_bool_expr(int i) {
  if (i & 1) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return i & 1;$}}

bool negated_if_implicit_bool_expr(int i) {
  if (i - 1) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return !\(i - 1\);$}}

bool implicit_int(int i) {
  if (i) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}  return i;{{$}}

bool explicit_bool(bool b) {
  if (b) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}  return b;{{$}}

bool negated_explicit_bool(bool b) {
  if (!b) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return !b;$}}

bool bitwise_complement_conversion(int i) {
  if (~i) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return ~i;$}}

bool logical_or(bool a, bool b) {
  if (a || b) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return a \|\| b;$}}

bool logical_and(bool a, bool b) {
  if (a && b) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return a && b;$}}

void ternary_integer_condition(int i) {
  bool b = i ? true : false;
}
// CHECK-MESSAGES: :[[@LINE-2]]:16: warning: redundant boolean literal in ternary expression result
// CHECK-FIXES: {{^  bool b = i;$}}

bool non_null_pointer_condition(int *p1) {
  if (p1) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return p1;$}}

bool null_pointer_condition(int *p2) {
  if (!p2) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return !p2;$}}

bool negated_non_null_pointer_condition(int *p3) {
  if (p3) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return !p3;$}}

bool negated_null_pointer_condition(int *p4) {
  if (!p4) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return p4;$}}

bool comments_in_the_middle(bool b) {
  if (b) {
    return true;
  } else {
    // something wicked this way comes
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-6]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}  if (b) {
// CHECK-FIXES: // something wicked this way comes{{$}}

bool preprocessor_in_the_middle(bool b) {
  if (b) {
    return true;
  } else {
#define SOMETHING_WICKED false
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-6]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^}}  if (b) {
// CHECK-FIXES: {{^}}#define SOMETHING_WICKED false

bool integer_not_zero(int i) {
  if (i) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: redundant boolean literal in conditional return
// CHECK-FIXES: {{^  return !i;$}}
