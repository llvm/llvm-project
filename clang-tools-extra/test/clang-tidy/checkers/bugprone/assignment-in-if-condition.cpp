// RUN: %check_clang_tidy %s bugprone-assignment-in-if-condition %t

void f(int arg) {
  int f = 3;
  if ((f = arg) || (f == (arg + 1)))
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

void f1(int arg) {
  int f = 3;
  if ((f == arg) || (f = (arg + 1)))
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

void f2(int arg) {
  int f = 3;
  if (f = arg)
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

volatile int v = 32;

void f3(int arg) {
  int f = 3;
  if ((f == arg) || ((arg + 6 < f) && (f = v)))
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  }
}

void f4(int arg) {
  int f = 3;
  if ((f == arg) || ((arg + 6 < f) && ((f = v) || (f < 8))))
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 5;
  } else if ((arg + 8 < f) && ((f = v) || (f < 8)))
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 7;
  }
  if ((arg == 3) || (bo = 6))
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: an assignment within an 'if' condition is bug-prone [bugprone-assignment-in-if-condition]
  {
    f = 8;
  }
  v = f;
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
