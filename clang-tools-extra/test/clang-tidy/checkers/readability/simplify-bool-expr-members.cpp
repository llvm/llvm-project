// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t

class A {
public:
  int m;
};

struct S {
  S() : m_ar(s_a) {}

  void operator_equals();
  void operator_or();
  void operator_and();
  void ternary_operator();
  void operator_not_equal();
  void simple_conditional_assignment_statements();
  void complex_conditional_assignment_statements();
  void chained_conditional_assignment();
  bool non_null_pointer_condition();
  bool null_pointer_condition();
  bool negated_non_null_pointer_condition();
  bool negated_null_pointer_condition();
  bool integer_not_zero();
  bool member_pointer_nullptr();
  bool integer_member_implicit_cast();
  bool expr_with_cleanups();

  bool m_b1 = false;
  bool m_b2 = false;
  bool m_b3 = false;
  bool m_b4 = false;
  int *m_p = nullptr;
  int A::*m_m = nullptr;
  int m_i = 0;
  A *m_a = nullptr;
  static A s_a;
  A &m_ar;
};

void S::operator_equals() {
  int i = 0;
  m_b1 = (i > 2);
  if (m_b1 == true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b1\) {$}}
    i = 5;
  } else {
    i = 6;
  }
  m_b2 = (i > 4);
  if (m_b2 == false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!m_b2\) {$}}
    i = 7;
  } else {
    i = 9;
  }
  m_b3 = (i > 6);
  if (true == m_b3) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b3\) {$}}
    i = 10;
  } else {
    i = 11;
  }
  m_b4 = (i > 8);
  if (false == m_b4) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!m_b4\) {$}}
    i = 12;
  } else {
    i = 13;
  }
}

void S::operator_or() {
  int i = 0;
  m_b1 = (i > 10);
  if (m_b1 || false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b1\) {$}}
    i = 14;
  } else {
    i = 15;
  }
  m_b2 = (i > 10);
  if (m_b2 || true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(true\) {$}}
    i = 16;
  } else {
    i = 17;
  }
  m_b3 = (i > 10);
  if (false || m_b3) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b3\) {$}}
    i = 18;
  } else {
    i = 19;
  }
  m_b4 = (i > 10);
  if (true || m_b4) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(true\) {$}}
    i = 20;
  } else {
    i = 21;
  }
}

void S::operator_and() {
  int i = 0;
  m_b1 = (i > 20);
  if (m_b1 && false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(false\) {$}}
    i = 22;
  } else {
    i = 23;
  }
  m_b2 = (i > 20);
  if (m_b2 && true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b2\) {$}}
    i = 24;
  } else {
    i = 25;
  }
  m_b3 = (i > 20);
  if (false && m_b3) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(false\) {$}}
    i = 26;
  } else {
    i = 27;
  }
  m_b4 = (i > 20);
  if (true && m_b4) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b4\) {$}}
    i = 28;
  } else {
    i = 29;
  }
}

void S::ternary_operator() {
  int i = 0;
  m_b1 = (i > 20) ? true : false;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{^  m_b1 = i > 20;$}}

  m_b2 = (i > 20) ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{^  m_b2 = i <= 20;$}}

  m_b3 = ((i > 20)) ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{^  m_b3 = i <= 20;$}}
}

void S::operator_not_equal() {
  int i = 0;
  m_b1 = (i > 20);
  if (false != m_b1) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b1\) {$}}
    i = 30;
  } else {
    i = 31;
  }
  m_b2 = (i > 20);
  if (true != m_b2) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!m_b2\) {$}}
    i = 32;
  } else {
    i = 33;
  }
  m_b3 = (i > 20);
  if (m_b3 != false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(m_b3\) {$}}
    i = 34;
  } else {
    i = 35;
  }
  m_b4 = (i > 20);
  if (m_b4 != true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!m_b4\) {$}}
    i = 36;
  } else {
    i = 37;
  }
}

void S::simple_conditional_assignment_statements() {
  if (m_i > 10)
    m_b1 = true;
  else
    m_b1 = false;
  bool bb = false;
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: {{^  }}m_b1 = m_i > 10;{{$}}
  // CHECK-FIXES: bool bb = false;

  if (m_i > 20)
    m_b2 = false;
  else
    m_b2 = true;
  bool c2 = false;
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: {{^  }}m_b2 = m_i <= 20;{{$}}
  // CHECK-FIXES: bool c2 = false;

  // Unchanged: different variables.
  if (m_i > 12)
    m_b1 = true;
  else
    m_b2 = false;

  // Unchanged: no else statement.
  if (m_i > 15)
    m_b3 = true;

  // Unchanged: not boolean assignment.
  int j;
  if (m_i > 17)
    j = 10;
  else
    j = 20;

  // Unchanged: different variables assigned.
  int k = 0;
  m_b4 = false;
  if (m_i > 10)
    m_b4 = true;
  else
    k = 10;
}

void S::complex_conditional_assignment_statements() {
  if (m_i > 30) {
    m_b1 = true;
  } else {
    m_b1 = false;
  }
  m_b1 = false;
  // CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: {{^  }}m_b1 = m_i > 30;{{$}}
  // CHECK-FIXES: m_b1 = false;

  if (m_i > 40) {
    m_b2 = false;
  } else {
    m_b2 = true;
  }
  m_b2 = false;
  // CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: {{^  }}m_b2 = m_i <= 40;{{$}}
  // CHECK-FIXES: m_b2 = false;
}

// Unchanged: chained return statements, but ChainedConditionalReturn not set.
void S::chained_conditional_assignment() {
  if (m_i < 0)
    m_b1 = true;
  else if (m_i < 10)
    m_b1 = false;
  else if (m_i > 20)
    m_b1 = true;
  else
    m_b1 = false;
}

bool S::non_null_pointer_condition() {
  if (m_p) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: return m_p != nullptr;{{$}}

bool S::null_pointer_condition() {
  if (!m_p) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: return m_p == nullptr;{{$}}

bool S::negated_non_null_pointer_condition() {
  if (m_p) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: return m_p == nullptr;{{$}}

bool S::negated_null_pointer_condition() {
  if (!m_p) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: return m_p != nullptr;{{$}}

bool S::integer_not_zero() {
  if (m_i) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: {{^}}  return m_i == 0;{{$}}

bool S::member_pointer_nullptr() {
  if (m_m) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: return m_m != nullptr;{{$}}

bool S::integer_member_implicit_cast() {
  if (m_a->m) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: return m_a->m != 0;{{$}}

bool operator!=(const A &, const A &) { return false; }
bool S::expr_with_cleanups() {
  if (m_ar != (A)m_ar)
    return false;

  return true;
}
// CHECK-MESSAGES: :[[@LINE-4]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: m_ar == (A)m_ar;{{$}}
