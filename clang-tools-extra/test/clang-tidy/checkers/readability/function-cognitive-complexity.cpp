// RUN: %check_clang_tidy %s readability-function-cognitive-complexity %t -- -config='{CheckOptions: {readability-function-cognitive-complexity.Threshold: 0}}' -- -std=c++11 -fblocks -fexceptions -w

// any function should be checked.

extern int ext_func(int x = 0);

int some_func(int x = 0);

static int some_other_func(int x = 0) {}

template<typename T> void some_templ_func(T x = 0) {}

class SomeClass {
public:
  int *begin(int x = 0);
  int *end(int x = 0);
  static int func(int x = 0);
  template<typename T> void some_templ_func(T x = 0) {}
  SomeClass() = default;
  SomeClass(SomeClass&) = delete;
};

// nothing ever decreases cognitive complexity, so we can check all the things
// in one go. none of the following should increase cognitive complexity:
void unittest_false() {
  {};
  ext_func();
  some_func();
  some_other_func();
  some_templ_func<int>();
  some_templ_func<bool>();
  SomeClass::func();
  SomeClass C;
  C.some_templ_func<int>();
  C.some_templ_func<bool>();
  C.func();
  C.end();
  int i = some_func();
  i = i;
  i++;
  --i;
  i < 0;
  int j = 0 ?: 1;
  auto k = new int;
  delete k;
  throw i;
  {
    throw i;
  }
end:
  return;
}

#if 1
#define CC100
#else
// this macro has cognitive complexity of 100.
// it is needed to be able to compare the testcases with the
// reference Sonar implementation. please place it right after the first
// CHECK-NOTES in each function
#define CC100 if(1){if(1){if(1){if(1){if(1){if(1){if(1){if(1){if(1){if(1){if(1){if(1){if(1){}}}}}if(1){}}}}}}}}}
#endif

//----------------------------------------------------------------------------//
//------------------------------ B1. Increments ------------------------------//
//----------------------------------------------------------------------------//
// Check that every thing listed in B1 of the specification does indeed       //
// recieve the base increment, and that not-body does not increase nesting    //
//----------------------------------------------------------------------------//

// break does not increase cognitive complexity.
// only  break LABEL  does, but it is unavailable in C or C++

// continue does not increase cognitive complexity.
// only  continue LABEL  does, but it is unavailable in C or C++

void unittest_b1_00() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_00' has cognitive complexity of 33 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:9: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

    if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:11: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    } else if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:12: note: +1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:18: note: +3, including nesting penalty of 2, nesting level increased to 3{{$}}
    } else {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, nesting level increased to 2{{$}}
    }
  } else if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:10: note: +1, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:16: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}

    if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:11: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    } else if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:12: note: +1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:18: note: +3, including nesting penalty of 2, nesting level increased to 3{{$}}
    } else {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, nesting level increased to 2{{$}}
    }
  } else {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, nesting level increased to 1{{$}}

    if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:11: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    } else if (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:12: note: +1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:18: note: +3, including nesting penalty of 2, nesting level increased to 3{{$}}
    } else {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b1_01() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_01' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  int i = (1 ? 1 : 0) ? 1 : 0;
// CHECK-NOTES: :[[@LINE-1]]:23: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:14: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
}

void unittest_b1_02(int x) {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_02' has cognitive complexity of 9 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  switch (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:13: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
  case -1:
    return;
  case 1 ? 1 : 0:
// CHECK-NOTES: :[[@LINE-1]]:10: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    return;
  case (1 ? 2 : 0) ... (1 ? 3 : 0):
// CHECK-NOTES: :[[@LINE-1]]:11: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-2]]:27: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    return;
  default:
    break;
  }
}

void unittest_b1_03(int x) {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_03' has cognitive complexity of 7 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  for (x = 1 ? 1 : 0; x < (1 ? 1 : 0); x += 1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:14: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-3]]:30: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-4]]:47: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    break;
    continue;
  }
}

void unittest_b1_04() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_04' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  SomeClass C;
  for (int i : (1 ? C : C)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:19: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    break;
    continue;
  }
}

void unittest_b1_05() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_05' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  while (1 ? 1 : 0) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:12: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    break;
    continue;
  }
}

void unittest_b1_06() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_06' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  do {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    break;
    continue;
  } while (1 ? 1 : 0);
// CHECK-NOTES: :[[@LINE-1]]:14: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
}

void unittest_b1_07() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_07' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  try {
  } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  }
}

void unittest_b1_08_00() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_08_00' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  goto end;
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1{{$}}
end:
  return;
}

void unittest_b1_08_01() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_08_01' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  void *ptr = &&end;
  goto *ptr;
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1{{$}}
end:
  return;
}

void unittest_b1_09_00() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_09_00' has cognitive complexity of 34 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if(1 && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
  }
  if(1 && 1 && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:13: note: +1{{$}}
  }
  if((1 && 1) && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:15: note: +1{{$}}
  }
  if(1 && (1 && 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
  }

  if(1 && 1 || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:13: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:8: note: +1{{$}}
  }
  if((1 && 1) || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:15: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:9: note: +1{{$}}
  }
  if(1 && (1 || 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:14: note: +1{{$}}
  }

  if(1 || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
  }
  if(1 || 1 || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:13: note: +1{{$}}
  }
  if((1 || 1) || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:15: note: +1{{$}}
  }
  if(1 || (1 || 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
  }

  if(1 || 1 && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:13: note: +1{{$}}
  }
  if((1 || 1) && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:15: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:9: note: +1{{$}}
  }
  if(1 || (1 && 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:14: note: +1{{$}}
  }
}

void unittest_b1_09_01() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_09_01' has cognitive complexity of 40 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if(1 && some_func(1 && 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
  if(1 && some_func(1 || 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
  if(1 || some_func(1 || 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
  if(1 || some_func(1 && 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }

  if(1 && some_func(1 && 1) && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
  if(1 && some_func(1 || 1) && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
  if(1 || some_func(1 || 1) && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-4]]:23: note: +1{{$}}
  }
  if(1 || some_func(1 && 1) && 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-4]]:23: note: +1{{$}}
  }

  if(1 && some_func(1 && 1) || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-4]]:23: note: +1{{$}}
  }
  if(1 && some_func(1 || 1) || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-4]]:23: note: +1{{$}}
  }
  if(1 || some_func(1 || 1) || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
  if(1 || some_func(1 && 1) || 1) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:29: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:23: note: +1{{$}}
  }
}

void unittest_b1_09_02() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b1_09_02' has cognitive complexity of 12 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if(1 && SomeClass::func(1 && 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:29: note: +1{{$}}
  }
  if(1 && SomeClass::func(1 || 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:29: note: +1{{$}}
  }
  if(1 || SomeClass::func(1 || 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:29: note: +1{{$}}
  }
  if(1 || SomeClass::func(1 && 1)) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:8: note: +1{{$}}
// CHECK-NOTES: :[[@LINE-3]]:29: note: +1{{$}}
  }
}

// FIXME: each method in a recursion cycle

//----------------------------------------------------------------------------//
//---------------------------- B2. Nesting lebel -----------------------------//
//----------------------------------------------------------------------------//
// Check that every thing listed in B2 of the specification does indeed       //
// increase the nesting level                                                 //
//----------------------------------------------------------------------------//

void unittest_b2_00() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_00' has cognitive complexity of 9 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  } else if (true) {
// CHECK-NOTES: :[[@LINE-1]]:10: note: +1, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  } else {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b2_01() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_01' has cognitive complexity of 5 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  int i = 1 ? (1 ? 1 : 0) : (1 ? 1 : 0);
// CHECK-NOTES: :[[@LINE-1]]:13: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
// CHECK-NOTES: :[[@LINE-2]]:18: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
// CHECK-NOTES: :[[@LINE-3]]:32: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
}

void unittest_b2_02(int x) {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_02' has cognitive complexity of 5 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  switch (x) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  case -1:
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
    return;
  default:
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
    return;
  }
}

void unittest_b2_03() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_03' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  for (;;) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b2_04() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_04' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  SomeClass C;
  for (int i : C) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b2_05() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_05' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  while (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b2_06() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_06' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  do {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  } while (true);
}

void unittest_b2_07() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_07' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  try {
  } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if(true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b2_08_00() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_08_00' has cognitive complexity of 10 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  class X {
    X() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    X &operator=(const X &other) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    ~X() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    void Y() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    static void Z() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

// CHECK-NOTES: :[[@LINE-45]]:5: warning: function 'X' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-42]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-39]]:8: warning: function 'operator=' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-36]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-33]]:5: warning: function '~X' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-30]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-27]]:10: warning: function 'Y' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-24]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-21]]:17: warning: function 'Z' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-18]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  };
}

void unittest_b2_08_01() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_08_01' has cognitive complexity of 10 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  struct X {
    X() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    X &operator=(const X &other) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    ~X() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    void Y() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

    static void Z() {
// CHECK-NOTES: :[[@LINE-1]]:5: note: nesting level increased to 1{{$}}
      CC100;

      if (true) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
      }
    }

// CHECK-NOTES: :[[@LINE-45]]:5: warning: function 'X' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-42]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-39]]:8: warning: function 'operator=' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-36]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-33]]:5: warning: function '~X' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-30]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-27]]:10: warning: function 'Y' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-24]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}

// CHECK-NOTES: :[[@LINE-21]]:17: warning: function 'Z' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-18]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  };
}

void unittest_b2_08_02() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_08_02' has cognitive complexity of 2 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  auto fun = []() {
// CHECK-NOTES: :[[@LINE-1]]:14: note: nesting level increased to 1{{$}}
    if (true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  };
// CHECK-NOTES: :[[@LINE-6]]:14: warning: lambda has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-5]]:5: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
}

void unittest_b2_09() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_09' has cognitive complexity of 2 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  ({
// CHECK-NOTES: :[[@LINE-1]]:3: note: nesting level increased to 1{{$}}
    if (true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  });
}

void unittest_b2_10() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b2_10' has cognitive complexity of 2 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  void (^foo)(void) = ^(void) {
// CHECK-NOTES: :[[@LINE-1]]:23: note: nesting level increased to 1{{$}}
    if (true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  };
}

//----------------------------------------------------------------------------//
//-------------------------- B3. Nesting increments --------------------------//
//----------------------------------------------------------------------------//
// Check that every thing listed in B3 of the specification does indeed       //
// recieve the penalty of the current nesting level                           //
//----------------------------------------------------------------------------//

void unittest_b3_00() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_00' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    if (true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b3_01() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_01' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    int i = 1 ? 1 : 0;
// CHECK-NOTES: :[[@LINE-1]]:15: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
  }
}

void unittest_b3_02(int x) {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_02' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    switch (x) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    case -1:
      return;
    default:
      return;
    }
  }
}

void unittest_b3_03() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_03' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    for (;;) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b3_04() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_04' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    SomeClass C;
    for (int i : C) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b3_05() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_05' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    while (true) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

void unittest_b3_06() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_06' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    do {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    } while (true);
  }
}

void unittest_b3_07() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'unittest_b3_07' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  if (true) {
// CHECK-NOTES: :[[@LINE-1]]:3: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +2, including nesting penalty of 1, nesting level increased to 2{{$}}
    }
  }
}

//----------------------------------------------------------------------------//
// Check that functions are being checked                                     //
//----------------------------------------------------------------------------//

class CheckClass {
  CheckClass(int x) {
// CHECK-NOTES: :[[@LINE-1]]:3: warning: function 'CheckClass' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void PrivateMemberFunction() {
// CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'PrivateMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void PrivateConstMemberFunction() const {
// CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'PrivateConstMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  static void PrivateStaticMemberFunction() {
// CHECK-NOTES: :[[@LINE-1]]:15: warning: function 'PrivateStaticMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

public:
  CheckClass() {
// CHECK-NOTES: :[[@LINE-1]]:3: warning: function 'CheckClass' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  operator bool() const {
// CHECK-NOTES: :[[@LINE-1]]:3: warning: function 'operator bool' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  ~CheckClass() {
// CHECK-NOTES: :[[@LINE-1]]:3: warning: function '~CheckClass' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void PublicMemberFunction() {
// CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'PublicMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void PublicConstMemberFunction() const {
// CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'PublicConstMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  static void PublicStaticMemberFunction() {
// CHECK-NOTES: :[[@LINE-1]]:15: warning: function 'PublicStaticMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void PublicFunctionDefinition();

protected:
  CheckClass(bool b) {
// CHECK-NOTES: :[[@LINE-1]]:3: warning: function 'CheckClass' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void ProtectedMemberFunction() {
// CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'ProtectedMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  void ProtectedConstMemberFunction() const {
// CHECK-NOTES: :[[@LINE-1]]:8: warning: function 'ProtectedConstMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }

  static void ProtectedStaticMemberFunction() {
// CHECK-NOTES: :[[@LINE-1]]:15: warning: function 'ProtectedStaticMemberFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
    CC100;

    try {
    } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:7: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
    }
  }
};

void CheckClass::PublicFunctionDefinition() {
// CHECK-NOTES: :[[@LINE-1]]:18: warning: function 'PublicFunctionDefinition' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  try {
  } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  }
}

#define uglyfunctionmacro(name)                                                \
  void name() {                                                                \
    CC100;                                                                     \
                                                                               \
    if (true) {                                                                \
      try {                                                                    \
      } catch (...) {                                                          \
      }                                                                        \
    }                                                                          \
  }

uglyfunctionmacro(MacroFunction)
// CHECK-NOTES: :[[@LINE-1]]:19: warning: function 'MacroFunction' has cognitive complexity of 3 (threshold 0) [readability-function-cognitive-complexity]
// CHECK-NOTES: :[[@LINE-2]]:1: note: +1, including nesting penalty of 0, nesting level increased to 1
// CHECK-NOTES: :[[@LINE-10]]:5: note: expanded from macro 'uglyfunctionmacro'
// CHECK-NOTES: :[[@LINE-4]]:1: note: +2, including nesting penalty of 1, nesting level increased to 2
// CHECK-NOTES: :[[@LINE-10]]:9: note: expanded from macro 'uglyfunctionmacro'

template<typename T>
void templatedFunction() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'templatedFunction' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  try {
  } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  }
}

template<>
void templatedFunction<bool>() {
// CHECK-NOTES: :[[@LINE-1]]:6: warning: function 'templatedFunction<bool>' has cognitive complexity of 1 (threshold 0) [readability-function-cognitive-complexity]
  CC100;

  try {
  } catch (...) {
// CHECK-NOTES: :[[@LINE-1]]:5: note: +1, including nesting penalty of 0, nesting level increased to 1{{$}}
  }
}

template void templatedFunction<int>();

void functionThatCallsTemplatedFunctions() {
  templatedFunction<int>();

  templatedFunction<bool>();

  templatedFunction<char>();

  templatedFunction<void*>();
}

static void pr47779_dont_crash_on_weak() __attribute__((__weakref__("__pr47779_dont_crash_on_weak")));
