// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t -- -config="{CheckOptions: [{key: "readability-simplify-boolean-expr.ChainedConditionalReturn", value: true}]}" --

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
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
  // CHECK-FIXES:      {{^}}  } else if (i < 10) {{{$}}
  // CHECK-FIXES-NEXT: {{^}}    return false;{{$}}
  // CHECK-FIXES-NEXT: {{^}}  } else return i > 20;{{$}}
}

bool chained_conditional_return(int i) {
  if (i < 0)
    return true;
  else if (i < 10)
    return false;
  else if (i > 20)
    return true;
  else
    return false;
  // CHECK-MESSAGES: :[[@LINE-3]]:12: warning: {{.*}} in conditional return statement
  // CHECK-FIXES:      {{^}}  else if (i < 10)
  // CHECK-FIXES-NEXT: {{^}}    return false;
  // CHECK-FIXES-NEXT: {{^}}  else return i > 20;
}

bool chained_simple_if_return(int i) {
  if (i < 5)
    return true;
  if (i > 10)
    return true;
  return false;
}
// CHECK-MESSAGES: :[[@LINE-3]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: {{^}}bool chained_simple_if_return(int i) {{{$}}
// CHECK-FIXES: {{^}}  if (i < 5){{$}}
// CHECK-FIXES: {{^    return true;$}}
// CHECK-FIXES: {{^  return i > 10;$}}
// CHECK-FIXES: {{^}$}}

bool chained_simple_if_return_negated(int i) {
  if (i < 5)
    return false;
  if (i > 10)
    return false;
  return true;
}
// CHECK-MESSAGES: :[[@LINE-3]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: {{^}}bool chained_simple_if_return_negated(int i) {{{$}}
// CHECK-FIXES: {{^}}  if (i < 5){{$}}
// CHECK-FIXES: {{^    return false;$}}
// CHECK-FIXES: {{^  return i <= 10;$}}
// CHECK-FIXES: {{^}$}}

bool complex_chained_if_return_return(int i) {
  if (i < 5) {
    return true;
  }
  if (i > 10) {
    return true;
  }
  return false;
}
// CHECK-MESSAGES: :[[@LINE-4]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: {{^}}bool complex_chained_if_return_return(int i) {{{$}}
// CHECK-FIXES: {{^}}  if (i < 5) {{{$}}
// CHECK-FIXES: {{^}}    return true;{{$}}
// CHECK-FIXES: {{^}}  }{{$}}
// CHECK-FIXES: {{^  return i > 10;$}}
// CHECK-FIXES: {{^}$}}

bool complex_chained_if_return_return_negated(int i) {
  if (i < 5) {
    return false;
  }
  if (i > 10) {
    return false;
  }
  return true;
}
// CHECK-MESSAGES: :[[@LINE-4]]:12: warning: {{.*}} in conditional return
// CHECK-FIXES: {{^}}bool complex_chained_if_return_return_negated(int i) {{{$}}
// CHECK-FIXES: {{^}}  if (i < 5) {{{$}}
// CHECK-FIXES: {{^}}    return false;{{$}}
// CHECK-FIXES: {{^}}  }{{$}}
// CHECK-FIXES: {{^  return i <= 10;$}}
// CHECK-FIXES: {{^}$}}


bool PR57819(int x) {
  // False positive introduced in clang-tidy-15
  // Expect no warning here.
  if (x > 0)
    return false;
  else {
  }
  return true;
}
