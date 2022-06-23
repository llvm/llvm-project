// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t -- -config="{CheckOptions: [{key: "readability-simplify-boolean-expr.ChainedConditionalAssignment", value: true}]}" --

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
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: redundant boolean literal in conditional assignment [readability-simplify-boolean-expr]
  // CHECK-FIXES:      {{^}}  } else if (i < 10) {{{$}}
  // CHECK-FIXES-NEXT: {{^}}    b = false;{{$}}
  // CHECK-FIXES-NEXT: {{^}}  } else b = i > 20;{{$}}
}

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
  // CHECK-MESSAGES: :[[@LINE-3]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES:      {{^}}  else if (i < 10)
  // CHECK-FIXES-NEXT: {{^}}    b = false;
  // CHECK-FIXES-NEXT: {{^}}  else b = i > 20;
}
