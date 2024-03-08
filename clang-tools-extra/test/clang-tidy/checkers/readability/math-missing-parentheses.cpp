// RUN: %check_clang_tidy %s readability-math-missing-parentheses %t

// FIXME: Add something that triggers the check here.
void f(){
    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int a = 1 + (2 * 3);
    int a = 1 + 2 * 3; 

    int b = 1 + 2 + 3; // No warning

    int c = 1 * 2 * 3; // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int d = (1 + (2 * 3)) - (4 / 5);
    int d = 1 + 2 * 3 - 4 / 5;

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int e = (1 & (2 + 3)) | (4 * 5);
    int e = 1 & 2 + 3 | 4 * 5;

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int f = (1 * -2) + 4;
    int f = 1 * -2 + 4;

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int g = (((1 * 2) * 3) + 4) + 5;
    int g = 1 * 2 * 3 + 4 + 5;

    // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int h = (120 & (2 + 3)) | (22 * 5);
    int h = 120 & 2 + 3 | 22 * 5;

    int i = 1 & 2 & 3; // No warning

    int j = 1 | 2 | 3; // No warning

    int k = 1 ^ 2 ^ 3; // No warning

    // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: add parantheses to clarify the precedence of operations [readability-math-missing-parentheses]
    // CHECK-FIXES: int l = (1 + 2) ^ 3;
    int l = 1 + 2 ^ 3;
}
