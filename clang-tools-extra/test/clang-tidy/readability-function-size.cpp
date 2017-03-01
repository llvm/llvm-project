// RUN: %check_clang_tidy %s readability-function-size %t -- -config='{CheckOptions: [{key: readability-function-size.LineThreshold, value: 0}, {key: readability-function-size.StatementThreshold, value: 0}, {key: readability-function-size.BranchThreshold, value: 0}, {key: readability-function-size.ParameterThreshold, value: 5}]}' -- -std=c++11

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
