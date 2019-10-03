// RUN: %check_clang_tidy %s bugprone-infinite-loop %t -- -- -fexceptions

void simple_infinite_loop1() {
  int i = 0;
  int j = 0;
  while (i < 10) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }

  do {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    j++;
  } while (i < 10);

  for (i = 0; i < 10; ++j) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
  }
}

void simple_infinite_loop2() {
  int i = 0;
  int j = 0;
  int Limit = 10;
  while (i < Limit) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i, Limit) are updated in the loop body [bugprone-infinite-loop]
    j++;
  }

  do {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i, Limit) are updated in the loop body [bugprone-infinite-loop]
    j++;
  } while (i < Limit);

  for (i = 0; i < Limit; ++j) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i, Limit) are updated in the loop body [bugprone-infinite-loop]
  }
}

void simple_not_infinite1() {
  int i = 0;
  int Limit = 100;
  while (i < Limit) {
    // Not an error since 'Limit' is updated.
    Limit--;
  }
  do {
    Limit--;
  } while (i < Limit);

  for (i = 0; i < Limit; Limit--) {
  }
}

void simple_not_infinite2() {
  for (int i = 10; i-- > 0;) {
    // Not an error, since loop variable is modified in its condition part.
  }
}

int unknown_function();

void function_call() {
  int i = 0;
  while (i < unknown_function()) {
    // Not an error, since the function may return different values.
  }

  do {
    // Not an error, since the function may return different values.
  } while (i < unknown_function());

  for (i = 0; i < unknown_function();) {
    // Not an error, since the function may return different values.
  }
}

void escape_before1() {
  int i = 0;
  int Limit = 100;
  int *p = &i;
  while (i < Limit) {
    // Not an error, since *p is alias of i.
    (*p)++;
  }

  do {
    (*p)++;
  } while (i < Limit);

  for (i = 0; i < Limit; ++(*p)) {
  }
}

void escape_before2() {
  int i = 0;
  int Limit = 100;
  int &ii = i;
  while (i < Limit) {
    // Not an error, since ii is alias of i.
    ii++;
  }

  do {
    ii++;
  } while (i < Limit);

  for (i = 0; i < Limit; ++ii) {
  }
}

void escape_inside1() {
  int i = 0;
  int Limit = 100;
  int *p = &i;
  while (i < Limit) {
    // Not an error, since *p is alias of i.
    int *p = &i;
    (*p)++;
  }

  do {
    int *p = &i;
    (*p)++;
  } while (i < Limit);
}

void escape_inside2() {
  int i = 0;
  int Limit = 100;
  while (i < Limit) {
    // Not an error, since ii is alias of i.
    int &ii = i;
    ii++;
  }

  do {
    int &ii = i;
    ii++;
  } while (i < Limit);
}

void escape_after1() {
  int i = 0;
  int j = 0;
  int Limit = 10;

  while (i < Limit) {
    // False negative, but difficult to detect without CFG-based analysis
  }
  int *p = &i;
}

void escape_after2() {
  int i = 0;
  int j = 0;
  int Limit = 10;

  while (i < Limit) {
    // False negative, but difficult to detect without CFG-based analysis
  }
  int &ii = i;
}

int glob;

void global1(int &x) {
  int i = 0, Limit = 100;
  while (x < Limit) {
    // Not an error since 'x' can be an alias of 'glob'.
    glob++;
  }
}

void global2() {
  int i = 0, Limit = 100;
  while (glob < Limit) {
    // Since 'glob' is declared out of the function we do not warn.
    i++;
  }
}

struct X {
  int m;

  void change_m();

  void member_expr1(int i) {
    while (i < m) {
      // False negative: No warning, since skipping the case where a struct or
      // class can be found in its condition.
      ;
    }
  }

  void member_expr2(int i) {
    while (i < m) {
      --m;
    }
  }

  void member_expr3(int i) {
    while (i < m) {
      change_m();
    }
  }
};

void array_index() {
  int i = 0;
  int v[10];
  while (i < 10) {
    v[i++] = 0;
  }

  i = 0;
  do {
    v[i++] = 0;
  } while (i < 9);

  for (i = 0; i < 10;) {
    v[i++] = 0;
  }

  for (i = 0; i < 10; v[i++] = 0) {
  }
}

void no_loop_variable() {
  while (0)
    ;
}

void volatile_in_condition() {
  volatile int cond = 0;
  while (!cond) {
  }
}

namespace std {
template<typename T> class atomic {
  T val;
public:
  atomic(T v): val(v) {};
  operator T() { return val; };
};
}

void atomic_in_condition() {
  std::atomic<int> cond = 0;
  while (!cond) {
  }
}

void loop_exit1() {
  int i = 0;
  while (i) {
    if (unknown_function())
      break;
  }
}

void loop_exit2() {
  int i = 0;
  while (i) {
    if (unknown_function())
      return;
  }
}

void loop_exit3() {
  int i = 0;
  while (i) {
    if (unknown_function())
      goto end;
  }
 end:
  ;
}

void loop_exit4() {
  int i = 0;
  while (i) {
    if (unknown_function())
      throw 1;
  }
}

[[noreturn]] void exit(int);

void loop_exit5() {
  int i = 0;
  while (i) {
    if (unknown_function())
      exit(1);
  }
}

void loop_exit_in_lambda() {
  int i = 0;
  while (i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this loop is infinite; none of its condition variables (i) are updated in the loop body [bugprone-infinite-loop]
    auto l = []() { return 0; };
  }
}

void lambda_capture() {
  int i = 0;
  int Limit = 100;
  int *p = &i;
  while (i < Limit) {
    // Not an error, since i is captured by reference in a lambda.
    auto l = [&i]() { ++i; };
  }

  do {
    int *p = &i;
    (*p)++;
  } while (i < Limit);
}
