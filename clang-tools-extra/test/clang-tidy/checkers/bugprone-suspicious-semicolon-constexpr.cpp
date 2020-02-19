// RUN: %check_clang_tidy %s bugprone-suspicious-semicolon %t -- -- -std=c++17

void fail()
{
  int x = 0;
  if(x > 5); (void)x;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: potentially unintended semicolon [bugprone-suspicious-semicolon]
  // CHECK-FIXES: if(x > 5) (void)x;
}

template <int X>
int foo(int a) {
    if constexpr(X > 0) {
        return a;
    }
    return a + 1;
}

template <int X>
int foo2(int a) {
    // FIXME: diagnose the case below. See https://reviews.llvm.org/D46234
    // for details.
    if constexpr(X > 0);
        return a;
    return a + 1;
}

int main(void) {
    foo2<0>(1);
    return foo<0>(1);
}
