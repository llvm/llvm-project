// RUN: %check_clang_tidy %s readability-avoid-return-with-void-value %t

void f1();

void f2() {
    return f1();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement within a void function should not have a specified return value [readability-avoid-return-with-void-value]
}

void f3(bool b) {
    if (b) return f1();
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: return statement within a void function should not have a specified return value [readability-avoid-return-with-void-value]
    return f2();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement within a void function should not have a specified return value [readability-avoid-return-with-void-value]
}

template<class T>
T f4() {}

void f5() {
    return f4<void>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement within a void function should not have a specified return value [readability-avoid-return-with-void-value]
}

void f6() { return; }

int f7() { return 1; }

int f8() { return f7(); }

void f9() {
    return (void)f7();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: return statement within a void function should not have a specified return value [readability-avoid-return-with-void-value]
}