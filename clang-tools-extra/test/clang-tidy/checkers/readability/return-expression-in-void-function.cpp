// RUN: %check_clang_tidy %s readability-return-expression-in-void-function %t

void f1();

void f2() {
    return f1();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement in void function should not return a value [readability-return-expression-in-void-function]
}

void f3(bool b) {
    if (b) return f1();
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: return statement in void function should not return a value [readability-return-expression-in-void-function]
    return f2();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement in void function should not return a value [readability-return-expression-in-void-function]
}

template<class T>
T f4() {}

void f5() {
    return f4<void>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement in void function should not return a value [readability-return-expression-in-void-function]
}
