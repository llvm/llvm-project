// RUN: %check_clang_tidy %s readability-math-missing-parentheses %t

#define MACRO_AND &
#define MACRO_ADD +
#define MACRO_OR |
#define MACRO_MULTIPLY *
#define MACRO_XOR ^
#define MACRO_SUBTRACT -
#define MACRO_DIVIDE /

int foo(){
    return 5;
}

int bar(){
    return 4;
}

int sink(int);
#define FUN(ARG) (sink(ARG))
#define FUN2(ARG) sink((ARG))
#define FUN3(ARG) sink(ARG)
#define FUN4(ARG) sink(1 + ARG)
#define FUN5(ARG) sink(4 * ARG)

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
    //CHECK-MESSAGES: :[[@LINE+2]]:17: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int a = 1 + (2 * 3);
    int a = 1 + 2 * 3;

    int a_negative = 1 + (2 * 3); // No warning

    int b = 1 + 2 + 3; // No warning

    int c = 1 * 2 * 3; // No warning

    //CHECK-MESSAGES: :[[@LINE+3]]:17: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+2]]:25: warning: '/' has higher precedence than '-'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int d = 1 + (2 * 3) - (4 / 5);
    int d = 1 + 2 * 3 - 4 / 5;

    int d_negative = 1 + (2 * 3) - (4 / 5); // No warning

    //CHECK-MESSAGES: :[[@LINE+4]]:13: warning: '&' has higher precedence than '|'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+3]]:17: warning: '+' has higher precedence than '&'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+2]]:25: warning: '*' has higher precedence than '|'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int e = (1 & (2 + 3)) | (4 * 5);
    int e = 1 & 2 + 3 | 4 * 5;

    int e_negative = (1 & (2 + 3)) | (4 * 5); // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int f = (1 * -2) + 4;
    int f = 1 * -2 + 4;

    int f_negative = (1 * -2) + 4; // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int g = (1 * 2 * 3) + 4 + 5;
    int g = 1 * 2 * 3 + 4 + 5;

    int g_negative = (1 * 2 * 3) + 4 + 5; // No warning

    //CHECK-MESSAGES: :[[@LINE+4]]:13: warning: '&' has higher precedence than '|'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+3]]:19: warning: '+' has higher precedence than '&'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+2]]:27: warning: '*' has higher precedence than '|'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int h = (120 & (2 + 3)) | (22 * 5);
    int h = 120 & 2 + 3 | 22 * 5;

    int h_negative = (120 & (2 + 3)) | (22 * 5); // No warning

    int i = 1 & 2 & 3; // No warning

    int j = 1 | 2 | 3; // No warning

    int k = 1 ^ 2 ^ 3; // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: '+' has higher precedence than '^'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int l = (1 + 2) ^ 3;
    int l = 1 + 2 ^ 3;

    int l_negative = (1 + 2) ^ 3; // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int m = (2 * foo()) + bar();
    int m = 2 * foo() + bar();

    int m_negative = (2 * foo()) + bar(); // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:13: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int n = (1.05 * foo()) + double(bar());
    int n = 1.05 * foo() + double(bar());

    int n_negative = (1.05 * foo()) + double(bar()); // No warning

    //CHECK-MESSAGES: :[[@LINE+3]]:17: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int o = 1 + (obj.A * 3) + obj.B;
    fun obj;
    int o = 1 + obj.A * 3 + obj.B;

    int o_negative = 1 + (obj.A * 3) + obj.B; // No warning

    //CHECK-MESSAGES: :[[@LINE+2]]:18: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int p = 1U + (2 * 3);
    int p = 1U + 2 * 3;

    int p_negative = 1U + (2 * 3); // No warning

    //CHECK-MESSAGES: :[[@LINE+7]]:13: warning: '+' has higher precedence than '|'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+6]]:25: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+5]]:53: warning: '&' has higher precedence than '^'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+4]]:53: warning: '^' has higher precedence than '|'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+3]]:77: warning: '-' has higher precedence than '^'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-MESSAGES: :[[@LINE+2]]:94: warning: '/' has higher precedence than '-'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: int q = (1 MACRO_ADD (2 MACRO_MULTIPLY 3)) MACRO_OR ((4 MACRO_AND 5) MACRO_XOR (6 MACRO_SUBTRACT (7 MACRO_DIVIDE 8)));
    int q = 1 MACRO_ADD 2 MACRO_MULTIPLY 3 MACRO_OR 4 MACRO_AND 5 MACRO_XOR 6 MACRO_SUBTRACT 7 MACRO_DIVIDE 8;

    //CHECK-MESSAGES: :[[@LINE+1]]:21: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    int r = FUN(0 + 1 * 2);

    //CHECK-MESSAGES: :[[@LINE+1]]:22: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    int s = FUN2(0 + 1 * 2);

    //CHECK-MESSAGES: :[[@LINE+1]]:22: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    int t = FUN3(0 + 1 * 2);

    //CHECK-MESSAGES: :[[@LINE+1]]:18: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    int u = FUN4(1 * 2);

    //CHECK-MESSAGES: :[[@LINE+1]]:13: warning: '*' has higher precedence than '+'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    int v = FUN5(0 + 1);
}

namespace PR92516 {
  void f(int i) {
    int j, k;
    for (j = i + 1, k = 0; j < 1; ++j) {}
  }

  void f2(int i) {
    int j;
    for (j = i + 1; j < 1; ++j) {}
  }

  void f3(int i) {
    int j;
    for (j = i + 1, 2; j < 1; ++j) {}
  }
}

namespace PR141249 {
  void AssignAsParentBinOp(int* netChange, int* nums, int k, int i) {
    //CHECK-MESSAGES: :[[@LINE+2]]:30: warning: '-' has higher precedence than '^'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
    //CHECK-FIXES: netChange[i] = nums[i] ^ (k - nums[i]);
    netChange[i] = nums[i] ^ k - nums[i];
  }
}

void CompareAsParentBinOp(int b) {
  //CHECK-MESSAGES: :[[@LINE+2]]:12: warning: '*' has higher precedence than '-'; add parentheses to explicitly specify the order of operations [readability-math-missing-parentheses]
  //CHECK-FIXES: if (b == (1 * 2) - 3)   {
  if (b == 1 * 2 - 3)   {

  }
}
