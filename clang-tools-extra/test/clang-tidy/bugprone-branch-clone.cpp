// RUN: %check_clang_tidy %s bugprone-branch-clone %t -- -- -fno-delayed-template-parsing

void test_basic1(int in, int &out) {
  if (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  else
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    out++;

  out++;
}

void test_basic2(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  }
  else {
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    out++;
  }

  out++;
}

void test_basic3(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  }
  else
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    out++;

  out++;
}

void test_basic4(int in, int &out) {
  if (in > 77) {
    out--;
  }
  else {
    out++;
  }
}

void test_basic5(int in, int &out) {
  if (in > 77) {
    out++;
  }
  else {
    out++;
    out++;
  }
}

void test_basic6(int in, int &out) {
  if (in > 77) {
    out++;
  }
  else {
    out++, out++;
  }
}

void test_basic7(int in, int &out) {
  if (in > 77) {
    out++;
    out++;
  }
  else
    out++;

  out++;
}

void test_basic8(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
    out++;
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    out++;
    out++;
  }

  if (in % 2)
    out++;
}

void test_basic9(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    if (in % 2)
      out++;
    else
      out--;
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    if (in % 2)
      out++;
    else
      out--;
  }
}

// If we remove the braces from the previous example, the check recognizes it
// as an `else if`.
void test_basic10(int in, int &out) {
  if (in > 77)
    if (in % 2)
      out++;
    else
      out--;
  else
    if (in % 2)
      out++;
    else
      out--;

}

void test_basic11(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    if (in % 2)
      out++;
    else
      out--;
    if (in % 3)
      out++;
    else
      out--;
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    if (in % 2)
      out++;
    else
      out--;
    if (in % 3)
      out++;
    else
      out--;
  }
}

void test_basic12(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
  }
}

void test_basic13(int in, int &out) {
  if (in > 77) {
    // Empty compound statement is not identical to null statement.
  } else;
}

// We use a comparison that ignores redundant parentheses:
void test_basic14(int in, int &out) {
  if (in > 77)
    out += 2;
  else
    (out) += (2);
}

void test_basic15(int in, int &out) {
  if (in > 77)
    ((out += 2));
  else
    out += 2;
}

// ..but does not apply additional simplifications:
void test_basic16(int in, int &out) {
  if (in > 77)
    out += 2;
  else
    out += 1 + 1;
}

// ..and does not forget important parentheses:
int test_basic17(int a, int b, int c, int mode) {
  if (mode>8)
    return (a + b) * c;
  else
    return a + b * c;
}

//=========--------------------==========//

#define PASTE_CODE(x) x

void test_macro1(int in, int &out) {
  PASTE_CODE(
    if (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: if with identical then and else branches [bugprone-branch-clone]
      out++;
    else
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
      out++;
  )

  out--;
}

void test_macro2(int in, int &out) {
  PASTE_CODE(
    if (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: if with identical then and else branches [bugprone-branch-clone]
      out++;
  )
  else
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    out++;
}

void test_macro3(int in, int &out) {
  if (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  PASTE_CODE(
    else
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
      out++;
  )
}

void test_macro4(int in, int &out) {
  if (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  else
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    PASTE_CODE(
      out++;
    )
}

void test_macro5(int in, int &out) {
  PASTE_CODE(if) (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  PASTE_CODE(else)
// CHECK-MESSAGES: :[[@LINE-1]]:14: note: else branch starts here
    out++;
}

#define OTHERWISE_INCREASE else out++

void test_macro6(int in, int &out) {
  if (in > 77)
      out++;
// CHECK-MESSAGES: :[[@LINE-2]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
  OTHERWISE_INCREASE;
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
// CHECK-MESSAGES: :[[@LINE-8]]:28: note: expanded from macro 'OTHERWISE_INCREASE'
}

#define COND_INCR(a, b, c) \
  do {                     \
    if ((a))               \
      (b)++;               \
    else                   \
      (c)++;               \
  } while (0)

void test_macro7(int in, int &out1, int &out2) {
  COND_INCR(in, out1, out1);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE-9]]:5: note: expanded from macro 'COND_INCR'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: else branch starts here
// CHECK-MESSAGES: :[[@LINE-9]]:5: note: expanded from macro 'COND_INCR'
}

void test_macro8(int in, int &out1, int &out2) {
  COND_INCR(in, out1, out2);
}

void test_macro9(int in, int &out1, int &out2) {
  COND_INCR(in, out2, out2);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE-21]]:5: note: expanded from macro 'COND_INCR'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: else branch starts here
// CHECK-MESSAGES: :[[@LINE-21]]:5: note: expanded from macro 'COND_INCR'
}

#define CONCAT(a, b) a##b

void test_macro10(int in, int &out) {
  CONCAT(i, f) (in > 77)
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    out++;
  CONCAT(el, se)
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    out++;
}

#define PROBLEM (-1)

int test_macro11(int count) {
  if (!count)
    return PROBLEM;
  else if (count == 13)
    return -1;
  else
    return count * 2;
}

#define IF if (
#define THEN ) {
#define ELSE } else {
#define END }

void test_macro12(int in, int &out) {
  IF in > 77 THEN
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE-8]]:12: note: expanded from macro 'IF'
    out++;
    out++;
  ELSE
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
// CHECK-MESSAGES: :[[@LINE-11]]:16: note: expanded from macro 'ELSE'
    out++;
    out++;
  END
}

// A hack for implementing a switch with no fallthrough:
#define SWITCH(x) switch (x) {
#define CASE(x) break; case (x):
#define DEFAULT break; default:

void test_macro13(int in, int &out) {
  SWITCH(in)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
    CASE(1)
      out++;
      out++;
    CASE(2)
      out++;
      out++;
    CASE(3)
      out++;
      out++;
// CHECK-MESSAGES: :[[@LINE-15]]:24: note: expanded from macro 'CASE'
// CHECK-MESSAGES: :[[@LINE+1]]:9: note: last of these clones ends here
    CASE(4)
      out++;
    CASE(5)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
    CASE(6)
      out--;
    CASE(7)
      out--;
// CHECK-MESSAGES: :[[@LINE-25]]:24: note: expanded from macro 'CASE'
// CHECK-MESSAGES: :[[@LINE+2]]:9: note: last of these clones ends here
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
    CASE(8)
      out++;
      out++;
    CASE(9)
      out++;
      out++;
// CHECK-MESSAGES: :[[@LINE-34]]:24: note: expanded from macro 'CASE'
// CHECK-MESSAGES: :[[@LINE+2]]:12: note: last of these clones ends here
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
    DEFAULT
      out--;
      out--;
    CASE(10)
      out--;
      out--;
// CHECK-MESSAGES: :[[@LINE-42]]:24: note: expanded from macro 'DEFAULT'
// CHECK-MESSAGES: :[[@LINE+1]]:9: note: last of these clones ends here
    CASE(12)
      out++;
    CASE(13)
      out++;
  END
}

//=========--------------------==========//

void test_chain1(int in, int &out) {
  if (in > 77)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: end of the original
  else if (in > 55)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 1 starts here
    out++;

  out++;
}

void test_chain2(int in, int &out) {
  if (in > 77)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: end of the original
  else if (in > 55)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 1 starts here
    out++;
  else if (in > 42)
    out--;
  else if (in > 28)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 2 starts here
    out++;
  else if (in > 12) {
    out++;
    out *= 7;
  } else if (in > 7) {
// CHECK-MESSAGES: :[[@LINE-1]]:22: note: clone 3 starts here
    out++;
  }
}

void test_chain3(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
    out++;
// CHECK-MESSAGES: :[[@LINE+1]]:4: note: end of the original
  } else if (in > 55) {
// CHECK-MESSAGES: :[[@LINE-1]]:23: note: clone 1 starts here
    out++;
    out++;
  } else if (in > 42)
    out--;
  else if (in > 28) {
// CHECK-MESSAGES: :[[@LINE-1]]:21: note: clone 2 starts here
    out++;
    out++;
  } else if (in > 12) {
    out++;
    out++;
    out++;
    out *= 7;
  } else if (in > 7) {
// CHECK-MESSAGES: :[[@LINE-1]]:22: note: clone 3 starts here
    out++;
    out++;
  }
}

// In this chain there are two clone families; notice that the checker
// describes all branches of the first one before mentioning the second one.
void test_chain4(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
    out++;
// CHECK-MESSAGES: :[[@LINE+1]]:4: note: end of the original
  } else if (in > 55) {
// CHECK-MESSAGES: :[[@LINE-1]]:23: note: clone 1 starts here
// CHECK-MESSAGES: :[[@LINE+8]]:21: note: clone 2 starts here
// CHECK-MESSAGES: :[[@LINE+15]]:22: note: clone 3 starts here
    out++;
    out++;
  } else if (in > 42)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out--;
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: end of the original
  else if (in > 28) {
    out++;
    out++;
  } else if (in > 12) {
    out++;
    out++;
    out++;
    out *= 7;
  } else if (in > 7) {
    out++;
    out++;
  } else if (in > -3) {
// CHECK-MESSAGES: :[[@LINE-1]]:23: note: clone 1 starts here
    out--;
  }
}

void test_chain5(int in, int &out) {
  if (in > 77)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: end of the original
  else if (in > 55)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 1 starts here
    out++;
  else if (in > 42)
    out--;
  else if (in > 28)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 2 starts here
    out++;
  else if (in > 12) {
    out++;
    out *= 7;
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: clone 3 starts here
    out++;
  }
}

void test_chain6(int in, int &out) {
  if (in > 77) {
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
    out++;
// CHECK-MESSAGES: :[[@LINE+1]]:4: note: end of the original
  } else if (in > 55) {
// CHECK-MESSAGES: :[[@LINE-1]]:23: note: clone 1 starts here
    out++;
    out++;
  } else if (in > 42)
    out--;
  else if (in > 28) {
// CHECK-MESSAGES: :[[@LINE-1]]:21: note: clone 2 starts here
    out++;
    out++;
  } else if (in > 12) {
    out++;
    out++;
    out++;
    out *= 7;
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: clone 3 starts here
    out++;
    out++;
  }
}

void test_nested(int a, int b, int c, int &out) {
  if (a > 5) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+27]]:5: note: else branch starts here
    if (b > 5) {
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: repeated branch in conditional chain [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+9]]:6: note: end of the original
// CHECK-MESSAGES: :[[@LINE+8]]:24: note: clone 1 starts here
// CHECK-MESSAGES: :[[@LINE+14]]:12: note: clone 2 starts here
      if (c > 5)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: if with identical then and else branches [bugprone-branch-clone]
        out++;
      else
// CHECK-MESSAGES: :[[@LINE-1]]:7: note: else branch starts here
        out++;
    } else if (b > 15) {
      if (c > 5)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: if with identical then and else branches [bugprone-branch-clone]
        out++;
      else
// CHECK-MESSAGES: :[[@LINE-1]]:7: note: else branch starts here
        out++;
    } else {
      if (c > 5)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: if with identical then and else branches [bugprone-branch-clone]
        out++;
      else
// CHECK-MESSAGES: :[[@LINE-1]]:7: note: else branch starts here
        out++;
    }
  } else {
    if (b > 5) {
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: repeated branch in conditional chain [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+9]]:6: note: end of the original
// CHECK-MESSAGES: :[[@LINE+8]]:24: note: clone 1 starts here
// CHECK-MESSAGES: :[[@LINE+14]]:12: note: clone 2 starts here
      if (c > 5)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: if with identical then and else branches [bugprone-branch-clone]
        out++;
      else
// CHECK-MESSAGES: :[[@LINE-1]]:7: note: else branch starts here
        out++;
    } else if (b > 15) {
      if (c > 5)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: if with identical then and else branches [bugprone-branch-clone]
        out++;
      else
// CHECK-MESSAGES: :[[@LINE-1]]:7: note: else branch starts here
        out++;
    } else {
      if (c > 5)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: if with identical then and else branches [bugprone-branch-clone]
        out++;
      else
// CHECK-MESSAGES: :[[@LINE-1]]:7: note: else branch starts here
        out++;
    }
  }
}

//=========--------------------==========//

template <class T>
void test_template_not_instantiated(const T &t) {
  int a;
  if (t)
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    a++;
  else
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    a++;
}

template <class T>
void test_template_instantiated(const T &t) {
  int a;
  if (t)
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    a++;
  else
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: else branch starts here
    a++;
}

template void test_template_instantiated<int>(const int &t);

template <class T>
void test_template2(T t, int a) {
  if (a) {
    T b(0);
    a += b;
  } else {
    int b(0);
    a += b;
  }
}

template void test_template2<int>(int t, int a);

template <class T>
void test_template3(T t, int a) {
  if (a) {
    T b(0);
    a += b;
  } else {
    int b(0);
    a += b;
  }
}

template void test_template3<short>(short t, int a);

template <class T>
void test_template_two_instances(T t, int &a) {
  if (a) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    a += int(t);
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    a += int(t);
  }
}

template void test_template_two_instances<short>(short t, int &a);
template void test_template_two_instances<long>(long t, int &a);

class C {
  int member;
  void inline_method(int arg) {
    if (arg)
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: if with identical then and else branches [bugprone-branch-clone]
      member = 3;
    else
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
      member = 3;
  }
  int other_method();
};

int C::other_method() {
  if (member) {
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: if with identical then and else branches [bugprone-branch-clone]
    return 8;
  } else {
// CHECK-MESSAGES: :[[@LINE-1]]:5: note: else branch starts here
    return 8;
  }
}

//=========--------------------==========//

int simple_switch(char ch) {
  switch (ch) {
// CHECK-MESSAGES: :[[@LINE+1]]:3: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
  case 'a':
    return 10;
  case 'A':
    return 10;
// CHECK-MESSAGES: :[[@LINE-1]]:14: note: last of these clones ends here
// CHECK-MESSAGES: :[[@LINE+1]]:3: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
  case 'b':
    return 11;
  case 'B':
    return 11;
// CHECK-MESSAGES: :[[@LINE-1]]:14: note: last of these clones ends here
// CHECK-MESSAGES: :[[@LINE+1]]:3: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
  case 'c':
    return 10;
  case 'C':
    return 10;
// CHECK-MESSAGES: :[[@LINE-1]]:14: note: last of these clones ends here
  default:
    return 0;
  }
}

int long_sequence_switch(char ch) {
  switch (ch) {
// CHECK-MESSAGES: :[[@LINE+1]]:3: warning: switch has 7 consecutive identical branches [bugprone-branch-clone]
  case 'a':
    return 10;
  case 'A':
    return 10;
  case 'b':
    return 10;
  case 'B':
    return 10;
  case 'c':
    return 10;
  case 'C':
    return 10;
  default:
    return 10;
// CHECK-MESSAGES: :[[@LINE-1]]:14: note: last of these clones ends here
  }
}

int nested_switch(int a, int b, int c) {
  switch (a) {
// CHECK-MESSAGES: :[[@LINE+2]]:3: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+114]]:6: note: last of these clones ends here
  case 1:
    switch (b) {
// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+33]]:8: note: last of these clones ends here
    case 1:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    case 2:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    default:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    }
  case 2:
    switch (b) {
// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+33]]:8: note: last of these clones ends here
    case 1:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    case 2:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    default:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    }
  default:
    switch (b) {
// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
// CHECK-MESSAGES: :[[@LINE+33]]:8: note: last of these clones ends here
    case 1:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    case 2:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    default:
      switch (c) {
// CHECK-MESSAGES: :[[@LINE+1]]:7: warning: switch has 3 consecutive identical branches [bugprone-branch-clone]
      case 1:
        return 42;
      case 2:
        return 42;
      default:
        return 42;
// CHECK-MESSAGES: :[[@LINE-1]]:18: note: last of these clones ends here
      }
    }
  }
}

//=========--------------------==========//

// This should not produce warnings, as in switch statements we only report
// identical branches when they are consecutive. Also note that a branch
// terminated by a break is different from a branch terminated by the end of
// the switch statement.
int interleaved_cases(int a, int b) {
  switch (a) {
  case 3:
  case 4:
    b = 2;
    break;
  case 5:
    b = 3;
    break;
  case 6:
    b = 2;
    break;
  case 7:
    if (b % 2) {
      b++;
    } else {
      b++;
      break;
    }
    b = 2;
    break;
  case 8:
    b = 2;
  case 9:
    b = 3;
    break;
  default:
    b = 3;
  }
  return b;
}


// A case: or default: is only considered to be the start of a branch if it is a direct child of the CompoundStmt forming the body of the switch
int buried_cases(int foo) {
  switch (foo) {
    {
    case 36:
      return 8;
    default:
      return 8;
    }
  }
}

// Here the `case 7:` is a child statement of the GotoLabelStmt, so the checker
// thinks that it is part of the `case 9:` branch. While this result is
// counterintuitve, mixing goto labels and switch statements in this fashion is
// pretty rare, so it does not deserve a special case in the checker code.
int decorated_cases(int z) {
  if (!(z % 777)) {
    goto lucky;
  }
  switch (z) {
// CHECK-MESSAGES: :[[@LINE+1]]:3: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
  case 1:
  case 2:
  case 3:
    z++;
    break;
  case 4:
  case 5:
    z++;
    break;
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: last of these clones ends here
  case 9:
    z++;
    break;
  lucky:
  case 7:
    z += 3;
    z *= 2;
    break;
  case 92:
    z += 3;
    z *= 2;
    break;
  default:
    z++;
  }
  return z + 92;
}

// The child of the switch statement is not neccessarily a compound statement,
// do not crash in this unusual case.
char no_real_body(int in, int &out) {
  switch (in)
  case 42:
    return 'A';

  if (in > 77)
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: repeated branch in conditional chain [bugprone-branch-clone]
    out++;
// CHECK-MESSAGES: :[[@LINE-1]]:10: note: end of the original
  else if (in > 55)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 1 starts here
    out++;
  else if (in > 34)
// CHECK-MESSAGES: :[[@LINE+1]]:5: note: clone 2 starts here
    out++;

  return '|';
}

// Duff's device [https://en.wikipedia.org/wiki/Duff's_device]
// The check does not try to distinguish branches in this sort of convoluted
// code, but it should avoid crashing.
void send(short *to, short *from, int count)
{
    int n = (count + 7) / 8;
    switch (count % 8) {
    case 0: do { *to = *from++;
    case 7:      *to = *from++;
    case 6:      *to = *from++;
    case 5:      *to = *from++;
    case 4:      *to = *from++;
    case 3:      *to = *from++;
    case 2:      *to = *from++;
    case 1:      *to = *from++;
            } while (--n > 0);
    }
}

//=========--------------------==========//

void ternary1(bool b, int &x) {
// CHECK-MESSAGES: :[[@LINE+1]]:6: warning: conditional operator with identical true and false expressions [bugprone-branch-clone]
  (b ? x : x) = 42;
}

int ternary2(bool b, int x) {
// CHECK-MESSAGES: :[[@LINE+1]]:12: warning: conditional operator with identical true and false expressions [bugprone-branch-clone]
  return b ? 42 : 42;
}

int ternary3(bool b, int x) {
  return b ? 42 : 43;
}

int ternary4(bool b, int x) {
  return b ? true ? 45 : 44 : false ? 45 : 44;
}

// We do not detect chains of conditional operators.
int ternary5(bool b1, bool b2, int x) {
  return b1 ? 42 : b2 ? 43 : 42;
}

#define SWITCH_WITH_LBRACE(b) switch (b) {
#define SEMICOLON_CASE_COLON(b)                                                \
  ;                                                                            \
  case b:
int d, e;
void dontCrash() {
  SWITCH_WITH_LBRACE(d)
// CHECK-MESSAGES: :[[@LINE+1]]:3: warning: switch has 2 consecutive identical branches [bugprone-branch-clone]
  SEMICOLON_CASE_COLON(1)
    e++;
    e++;
  SEMICOLON_CASE_COLON(2)
    e++;
    e++;
  // CHECK-MESSAGES: :[[@LINE-11]]:3: note: expanded from macro 'SEMICOLON_CASE_COLON'
  // CHECK-MESSAGES: :[[@LINE+1]]:23: note: last of these clones ends here
  SEMICOLON_CASE_COLON(3);
  }
}
