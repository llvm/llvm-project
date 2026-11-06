// RUN: %check_clang_tidy %s bugprone-assignment-in-if-condition %t

void f(int arg) {
  int f = 3;
  if ((f = arg) || (f == (arg + 1)))
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

void f1(int arg) {
  int f = 3;
  if ((f == arg) || (f = (arg + 1)))
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

void f2(int arg) {
  int f = 3;
  if (f = arg)
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

volatile int v = 32;

void f3(int arg) {
  int f = 3;
  if ((f == arg) || ((arg + 6 < f) && (f = v)))
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

void f4(int arg) {
  int f = 3;
  if ((f == arg) || ((arg + 6 < f) && ((f = v) || (f < 8))))
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  } else if ((arg + 8 < f) && ((f = v) || (f < 8)))
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 6;
  }
}

class BrokenOperator {
public:
  int d = 0;
  int operator=(const int &val) {
    d = val + 1;
    return d;
  }
};

void f5(int arg) {
  BrokenOperator bo;
  int f = 3;
  bo = f;
  if (bo.d == 3) {
    f = 6;
  }
  if (bo = 3)
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 7;
  }
  if ((arg == 3) || (bo = 6))
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 8;
  }
  v = f;
}

void f6(int arg) {
  int x;
  if (({ x = 5; }))
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

void f7(int arg) {
  if (({ int r; r = arg; (r = 1); }))
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

void f8(int arg) {
  if (({ label: arg = 1; }))
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

int get_value();

void f9(int arg) {
  if (({ [[clang::nomerge]] arg = get_value(); }))
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

void f10(int x, int y) {
  if (({ x = 1; ({ y = 2; }); }))
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

void f11(int x) {
  if (({ ({ x = 1; }); }))
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

void f12(int arg) {
  BrokenOperator bo1, bo2;
  if (({ bo1 = arg; ({ bo2 = arg; }); }))
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

void f13(int arg) {
  BrokenOperator bo;
  if (({ ({ bo = arg; }); }))
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
    return;
}

// Tests that shouldn't trigger warnings.
void awesome_f2(int arg) {
  int f = 3;
  if ((f == arg) || (f == (arg + 1))) {
    f = 5;
  }
}

void awesome_f3(int arg) {
  int f = 3;
  if (f == arg) {
    f = 5;
  }
}

void awesome_f4(int arg) {
  int f = 3;
  if ((f == arg) || ((arg + 6 < f) && ((f == v) || (f < 8)))) {
    f = 5;
  }
}

#define STMT_EXPR_MACRO(arg)                                                   \
  ({                                                                           \
    bool R;                                                                    \
    R = (arg) != 1;                                                            \
    R;                                                                         \
  })

void awesome_f5(int arg) {
  if (STMT_EXPR_MACRO(arg))
    return;
}

void awesome_f6(int arg) {
  if (({ int r; r = arg; r; }))
    return;
}

void awesome_f7(int x, int y) {
  if (({ x = 1; ({ y = 2; y; }); }))
    return;
}

void awesome_f8(int arg) {
  BrokenOperator bo;
  if (({ bo = arg; ({ bo = 2; bo.d; }); }))
    return;
}

template <typename Func> bool exec(Func F) { return F(); }

void lambda_if() {
  int X;
  if ([&X] {
        X = 5;
        return true;
      }()) {
  }

  if (exec([&] {
        X = 5;
        return true;
      })) {
  }
}
