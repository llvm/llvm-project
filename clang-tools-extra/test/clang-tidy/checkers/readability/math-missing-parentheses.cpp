// RUN: %check_clang_tidy %s readability-math-missing-parentheses %t

// FIXME: Add something that triggers the check here.

int foo(){
    return 5;
}

int bar(){
    return 4;
}

class fun{
public:  
    int A;
    double B;
    fun(){
        A = 5;
        B = 5.4;
    }
};

void f(){
    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int a = (1 + (2 * 3));
    int a = 1 + 2 * 3; 

    int b = 1 + 2 + 3; // No warning

    int c = 1 * 2 * 3; // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int d = ((1 + (2 * 3)) - (4 / 5));
    int d = 1 + 2 * 3 - 4 / 5;

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int e = ((1 & (2 + 3)) | (4 * 5));
    int e = 1 & 2 + 3 | 4 * 5;

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int f = ((1 * -2) + 4);
    int f = 1 * -2 + 4;

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int g = ((((1 * 2) * 3) + 4) + 5);
    int g = 1 * 2 * 3 + 4 + 5;

    // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int h = ((120 & (2 + 3)) | (22 * 5));
    int h = 120 & 2 + 3 | 22 * 5;

    int i = 1 & 2 & 3; // No warning

    int j = 1 | 2 | 3; // No warning

    int k = 1 ^ 2 ^ 3; // No warning

    // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int l = ((1 + 2) ^ 3);
    int l = 1 + 2 ^ 3;

    // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int m = ((2 * foo()) + bar());
    int m = 2 * foo() + bar();

    // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int n = ((1.05 * foo()) + double(bar()));
    int n = 1.05 * foo() + double(bar());

    // CHECK-MESSAGES: :[[@LINE+3]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int o = (1 + (obj.A * 3)) + obj.B;
    fun obj;
    int o = 1 + obj.A * 3 + obj.B; 
}