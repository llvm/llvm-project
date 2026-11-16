// RUN: %check_clang_tidy %s readability-redundant-control-flow %t
// RUN: %clang_cc1 -std=c++11 -fsyntax-only %t.cpp

void work(int);

void return_nested_scope() {
  {
    {
      work(0);
      return;
    }
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant return statement
// CHECK-FIXES-LABEL: void return_nested_scope() {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: work(0);
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void return_if(bool condition) {
  if (condition) {
    work(0);
    return;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: warning: redundant return statement
// CHECK-FIXES-LABEL: void return_if(bool condition) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: work(0);
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void return_if_else(bool condition) {
  if (condition) {
    work(0);
    return;
  } else {
    work(1);
    return;
  }
}
// CHECK-MESSAGES: :[[@LINE-6]]:5: warning: redundant return statement
// CHECK-MESSAGES: :[[@LINE-4]]:5: warning: redundant return statement
// CHECK-FIXES-LABEL: void return_if_else(bool condition) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: work(0);
// CHECK-FIXES-NEXT: } else {
// CHECK-FIXES-NEXT: work(1);
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void continue_nested_scope(int n) {
  for (int i = 0; i < n; ++i) {
    {
      {
        work(i);
        continue;
      }
    }
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: redundant continue statement
// CHECK-FIXES-LABEL: void continue_nested_scope(int n) {
// CHECK-FIXES-NEXT: for (int i = 0; i < n; ++i) {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: work(i);
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void continue_if(int n, bool condition) {
  for (int i = 0; i < n; ++i) {
    if (condition) {
      work(i);
      continue;
    }
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:7: warning: redundant continue statement
// CHECK-FIXES-LABEL: void continue_if(int n, bool condition) {
// CHECK-FIXES-NEXT: for (int i = 0; i < n; ++i) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: work(i);
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void continue_if_else(int n, bool condition) {
  for (int i = 0; i < n; ++i) {
    if (condition) {
      work(i);
      continue;
    } else {
      work(-i);
      continue;
    }
  }
}
// CHECK-MESSAGES: :[[@LINE-7]]:7: warning: redundant continue statement
// CHECK-MESSAGES: :[[@LINE-5]]:7: warning: redundant continue statement
// CHECK-FIXES-LABEL: void continue_if_else(int n, bool condition) {
// CHECK-FIXES-NEXT: for (int i = 0; i < n; ++i) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: work(i);
// CHECK-FIXES-NEXT: } else {
// CHECK-FIXES-NEXT: work(-i);
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void other_loops(bool condition) {
  while (condition) {
    if (condition) {
      continue;
    }
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: redundant continue statement
  do {
    {
      continue;
    }
  } while (condition);
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: redundant continue statement
  int values[] = {1, 2};
  for (int value : values) {
    if (value) {
      continue;
    }
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: redundant continue statement
}
// CHECK-FIXES-LABEL: void other_loops(bool condition) {
// CHECK-FIXES: while (condition) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }
// CHECK-FIXES: do {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: } while (condition);
// CHECK-FIXES: for (int value : values) {
// CHECK-FIXES-NEXT: if (value) {
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: }

void unbraced_returns(bool first, bool second) {
  if (first)
    return;
  else if (second)
    return;
  else
    return;
}
// CHECK-MESSAGES: :[[@LINE-6]]:5: warning: redundant return statement
// CHECK-MESSAGES: :[[@LINE-5]]:5: warning: redundant return statement
// CHECK-MESSAGES: :[[@LINE-4]]:5: warning: redundant return statement
// CHECK-FIXES-LABEL: void unbraced_returns(bool first, bool second) {
// CHECK-FIXES-NEXT: if (first)
// CHECK-FIXES-NEXT: ;
// CHECK-FIXES-NEXT: else if (second)
// CHECK-FIXES-NEXT: ;
// CHECK-FIXES-NEXT: else
// CHECK-FIXES-NEXT: ;
// CHECK-FIXES-NEXT: }

void unbraced_continue(int n) {
  for (int i = 0; i < n; ++i)
    if (i)
      continue;
}
// CHECK-MESSAGES: :[[@LINE-2]]:7: warning: redundant continue statement
// CHECK-FIXES-LABEL: void unbraced_continue(int n) {
// CHECK-FIXES-NEXT: for (int i = 0; i < n; ++i)
// CHECK-FIXES-NEXT: if (i)
// CHECK-FIXES-NEXT: ;
// CHECK-FIXES-NEXT: }

void non_tail_scope() {
  {
    return;
  }
  work(0);
}
// CHECK-FIXES-LABEL: void non_tail_scope() {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: return;
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: work(0);

void non_tail_branch(bool condition) {
  if (condition) {
    return;
  }
  work(0);
}
// CHECK-FIXES-LABEL: void non_tail_branch(bool condition) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: return;
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: work(0);

void return_in_loop(int n) {
  for (int i = 0; i < n; ++i) {
    if (i) {
      return;
    }
  }
}
// CHECK-FIXES-LABEL: void return_in_loop(int n) {
// CHECK-FIXES-NEXT: for (int i = 0; i < n; ++i) {
// CHECK-FIXES-NEXT: if (i) {
// CHECK-FIXES-NEXT: return;
// CHECK-FIXES-NEXT: }

void continue_before_statement(int n) {
  for (int i = 0; i < n; ++i) {
    {
      continue;
    }
    work(i);
  }
}
// CHECK-FIXES-LABEL: void continue_before_statement(int n) {
// CHECK-FIXES-NEXT: for (int i = 0; i < n; ++i) {
// CHECK-FIXES-NEXT: {
// CHECK-FIXES-NEXT: continue;
// CHECK-FIXES-NEXT: }
// CHECK-FIXES-NEXT: work(i);

void return_expression(bool condition) {
  if (condition) {
    return work(0);
  } else {
    return work(1);
  }
}
// CHECK-FIXES-LABEL: void return_expression(bool condition) {
// CHECK-FIXES-NEXT: if (condition) {
// CHECK-FIXES-NEXT: return work(0);
// CHECK-FIXES-NEXT: } else {
// CHECK-FIXES-NEXT: return work(1);

void switch_boundary(int n) {
  for (int i = 0; i < n; ++i) {
    switch (i) {
    case 0:
      continue;
    default:
      work(i);
    }
  }
}
// CHECK-FIXES-LABEL: void switch_boundary(int n) {
// CHECK-FIXES: case 0:
// CHECK-FIXES-NEXT: continue;
