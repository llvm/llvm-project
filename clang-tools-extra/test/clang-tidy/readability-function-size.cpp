// RUN: %check_clang_tidy %s readability-function-size %t -- -config='{CheckOptions: [{key: readability-function-size.LineThreshold, value: 0}, {key: readability-function-size.StatementThreshold, value: 0}, {key: readability-function-size.BranchThreshold, value: 0}, {key: readability-function-size.ParameterThreshold, value: 5}, {key: readability-function-size.NestingThreshold, value: 2}]}' -- -std=c++11

// Bad formatting is intentional, don't run clang-format over the whole file!

void foo1() {
}

void foo2() {;}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'foo2' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 1 statements (threshold 0)

void foo3() {
;

}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'foo3' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 1 statements (threshold 0)

void foo4(int i) { if (i) {} else; {}
}
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: function 'foo4' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 1 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 3 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 1 branches (threshold 0)

void foo5(int i) {for(;i;)while(i)
do;while(i);
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: function 'foo5' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 2 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 7 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 3 branches (threshold 0)

template <typename T> T foo6(T i) {return i;
}
int x = foo6(0);
// CHECK-MESSAGES: :[[@LINE-3]]:25: warning: function 'foo6' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-4]]:25: note: 1 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-5]]:25: note: 1 statements (threshold 0)

void foo7(int p1, int p2, int p3, int p4, int p5, int p6) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'foo7' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 1 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 6 parameters (threshold 5)

void bar1() { [](){;;;;;;;;;;;if(1){}}();


}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'bar1' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 14 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 1 branches (threshold 0)

void bar2() { class A { void barx() {;;} }; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'bar2' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 3 statements (threshold 0)
//
// CHECK-MESSAGES: :[[@LINE-4]]:30: warning: function 'barx' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-5]]:30: note: 2 statements (threshold 0)

#define macro() {int x; {int y; {int z;}}}

void baz0() { // 1
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'baz0' exceeds recommended size/complexity
  // CHECK-MESSAGES: :[[@LINE-2]]:6: note: 27 lines including whitespace and comments (threshold 0)
  // CHECK-MESSAGES: :[[@LINE-3]]:6: note: 9 statements (threshold 0)
  int a;
  { // 2
    int b;
    { // 3
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: nesting level 3 starts here (threshold 2)
      int c;
      { // 4
        int d;
      }
    }
  }
  { // 2
    int e;
  }
  { // 2
    { // 3
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: nesting level 3 starts here (threshold 2)
      int j;
    }
  }
  macro()
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: nesting level 3 starts here (threshold 2)
// CHECK-MESSAGES: :[[@LINE-28]]:25: note: expanded from macro 'macro'
}

// check that nested if's are not reported. this was broken initially
void nesting_if() { // 1
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'nesting_if' exceeds recommended size/complexity
  // CHECK-MESSAGES: :[[@LINE-2]]:6: note: 22 lines including whitespace and comments (threshold 0)
  // CHECK-MESSAGES: :[[@LINE-3]]:6: note: 18 statements (threshold 0)
  // CHECK-MESSAGES: :[[@LINE-4]]:6: note: 6 branches (threshold 0)
  if (true) { // 2
     int j;
  } else if (true) { // 2
     int j;
     if (true) { // 3
       // CHECK-MESSAGES: :[[@LINE-1]]:16: note: nesting level 3 starts here (threshold 2)
       int j;
     }
  } else if (true) { // 2
     int j;
     if (true) { // 3
       // CHECK-MESSAGES: :[[@LINE-1]]:16: note: nesting level 3 starts here (threshold 2)
       int j;
     }
  } else if (true) { // 2
     int j;
  }
}

// however this should warn
void bad_if_nesting() { // 1
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'bad_if_nesting' exceeds recommended size/complexity
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: 22 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: 12 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: 4 branches (threshold 0)
  if (true) {    // 2
    int j;
  } else { // 2
    if (true) { // 3
      // CHECK-MESSAGES: :[[@LINE-1]]:15: note: nesting level 3 starts here (threshold 2)
      int j;
    } else { // 3
      // CHECK-MESSAGES: :[[@LINE-1]]:12: note: nesting level 3 starts here (threshold 2)
      if (true) { // 4
        int j;
      } else { // 4
        if (true) { // 5
          int j;
        }
      }
    }
  }
}
