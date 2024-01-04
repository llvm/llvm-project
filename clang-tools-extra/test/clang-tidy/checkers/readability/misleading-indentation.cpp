// RUN: %check_clang_tidy %s readability-misleading-indentation %t -- -- -fno-delayed-template-parsing

void foo1();
void foo2();
void foo3();

#define E

#define BLOCK \
  if (cond1)  \
    foo1();   \
    foo2();

void f()
{
  bool cond1 = true;
  bool cond2 = true;

  if (cond1)
    if (cond2)
      foo1();
  else
    foo2();
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]

  if (cond1) {
    if (cond2)
      foo1();
  }
  else
    foo2();

  if (cond1)
    if (cond2)
      foo1();
    else
      foo2();

  if (cond2)
    foo1();
    foo2();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: misleading indentation: statement is indented too deeply [readability-misleading-indentation]
    // CHECK-MESSAGES: :[[@LINE-4]]:3: note: did you mean this line to be inside this 'if'
    foo2(); // No redundant warning.

  if (cond1)
  {
    foo1();
  }
    foo2();

  if (cond1)
    foo1();
  foo2();

  if (cond2)
    if (cond1) foo1(); else foo2();

  if (cond1) {
  } else {
  }

  if (cond1) {
  }
  else {
  }

  if (cond1)
  {
  }
  else
  {
  }

  if (cond1)
    {
    }
  else
    {
    }

  if(cond1) {
  }
  else if (cond2) {
  }
  else {
  }

  if(cond1) {
  }
  else if (cond2) {
  }
       else {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]

  if (cond1) {
    if (cond1) {
    }
    else if (cond2) {
    }
    else {
    }
    if (cond1) {
    } else if (cond2) {
    } else if (!cond2) {
    } else {
    }
  }
  else if (cond2) {
  }

  BLOCK

  if (cond1)
    foo1();
  else
    foo2();
  E foo3();
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]readability-misleading-indentation
}

void g(bool x) {
  if (x)
    #pragma unroll
    for (int k = 0; k < 1; ++k) {}

  #pragma unroll
  for (int k = 0; k < 1; ++k) {}
}

template<bool b>
void mustPass() {
  if constexpr (b) {
    foo1();
  } else {
    foo2();
  }
}

void mustPassNonTemplate() {
  constexpr unsigned Value = 1;
  if constexpr (Value == 0) {
    foo1();
  } else if constexpr (Value == 1) {
    foo2();
  } else {
    foo3();
  }
}

template<bool b>
void mustFail() {
  if constexpr (b) {
    foo1();
  }
    else {
      foo2();
      // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]
  }
}

void mustFailNonTemplate() {
  constexpr unsigned Value = 1;
  if constexpr (Value == 0) {
    foo1();
  }
    else {
  foo2();
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]
  }

  if constexpr (Value == 0)
    foo1();
    else
  foo2();
  // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]
}

template<bool b>
void mustFailNoInsta() {
  if constexpr (b) {
    foo1();
  }
    else {
      foo2();
      // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: different indentation for 'if' and corresponding 'else' [readability-misleading-indentation]
  }
}

template<bool b>
void mustPassNoInsta() {
  if constexpr (b) {
    foo1();
  }
  else {
    foo2();
  }
}

void call() {
  mustPass<true>();
  mustPass<false>();
  mustFail<true>();
  mustFail<false>();
}
