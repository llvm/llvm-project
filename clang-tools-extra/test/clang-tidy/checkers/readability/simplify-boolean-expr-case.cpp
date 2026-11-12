// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t

bool switch_stmt(int i, int j, bool b) {
  switch (i) {
  case 0:
    if (b == true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}

  case 1:
    if (b == false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(!b\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}

  case 2:
    if (b && true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}

  case 3:
    if (b && false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(false\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}

  case 4:
    if (b || true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(true\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
    // CHECK-FIXES-NEXT: {{break;}}

  case 5:
    if (b || false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}

  case 6:
    return i > 0 ? true : false;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}} in ternary expression result
    // CHECK-FIXES: {{return i > 0;}}

  case 7:
    return i > 0 ? false : true;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}} in ternary expression result
    // CHECK-FIXES: {{return i <= 0;}}

  case 8:
    if (true)
      j = 10;
    else
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in if statement condition
    // CHECK-FIXES:      {{j = 10;$}}
    // CHECK-FIXES-NEXT: {{break;$}}

  case 9:
    if (false)
      j = -20;
    else
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in if statement condition
    // CHECK-FIXES: {{j = 10;}}
    // CHECK-FIXES-NEXT: {{break;}}

  case 10:
    if (j > 10)
      return true;
    else
      return false;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} in conditional return statement
    // CHECK-FIXES: {{return j > 10;}}

  case 11:
    if (j > 10)
      return false;
    else
      return true;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} in conditional return statement
    // CHECK-FIXES: {{return j <= 10;}}

  case 12:
    if (j > 10)
      b = true;
    else
      b = false;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: {{.*}} in conditional assignment
    // CHECK-FIXES: {{b = j > 10;}}
    // CHECK-FIXES-NEXT: {{return b;}}

  case 13:
    if (j > 10)
      b = false;
    else
      b = true;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: {{.*}} in conditional assignment
    // CHECK-FIXES: {{b = j <= 10;}}
    // CHECK-FIXES-NEXT: {{return b;}}

  case 14:
    if (j > 10)
      return true;
    return false;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: {{.*}} in conditional return
    // FIXES: {{return j > 10;}}

  case 15:
    if (j > 10)
      return false;
    return true;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: {{.*}} in conditional return
    // FIXES: {{return j <= 10;}}

  case 16:
    if (j > 10)
      return true;
    return true;
    return false;

  case 17:
    if (j > 10)
      return false;
    return false;
    return true;

  case 100: {
    if (b == true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
  }

  case 101: {
    if (b == false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(!b\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}
  }

  case 102: {
    if (b && true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
  }

  case 103: {
    if (b && false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(false\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}
  }

  case 104: {
    if (b || true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(true\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
    // CHECK-FIXES-NEXT: {{break;}}
  }

  case 105: {
    if (b || false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}
  }

  case 106: {
    return i > 0 ? true : false;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}} in ternary expression result
    // CHECK-FIXES: {{return i > 0;}}
  }

  case 107: {
    return i > 0 ? false : true;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}} in ternary expression result
    // CHECK-FIXES: {{return i <= 0;}}
  }

  case 108: {
    if (true)
      j = 10;
    else
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in if statement condition
    // CHECK-FIXES:      {{j = 10;$}}
    // CHECK-FIXES-NEXT: {{break;$}}
  }

  case 109: {
    if (false)
      j = -20;
    else
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in if statement condition
    // CHECK-FIXES: {{j = 10;}}
    // CHECK-FIXES-NEXT: {{break;}}
  }

  case 110: {
    if (j > 10)
      return true;
    else
      return false;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} in conditional return statement
    // CHECK-FIXES: {{return j > 10;}}
  }

  case 111: {
    if (j > 10)
      return false;
    else
      return true;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} in conditional return statement
    // CHECK-FIXES: {{return j <= 10;}}
  }

  case 112: {
    if (j > 10)
      b = true;
    else
      b = false;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: {{.*}} in conditional assignment
    // CHECK-FIXES: {{b = j > 10;}}
    // CHECK-FIXES-NEXT: {{return b;}}
  }

  case 113: {
    if (j > 10)
      b = false;
    else
      b = true;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: {{.*}} in conditional assignment
    // CHECK-FIXES: {{b = j <= 10;}}
    // CHECK-FIXES-NEXT: {{return b;}}
  }

  case 114: {
    if (j > 10)
      return true;
    return false;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: {{.*}} in conditional return
    // CHECK-FIXES: {{return j > 10;}}
  }

  case 115: {
    if (j > 10)
      return false;
    return true;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: {{.*}} in conditional return
    // CHECK-FIXES: {{return j <= 10;}}
  }

  case 116: {
    return false;
    if (j > 10)
      return true;
  }

  case 117: {
    return true;
    if (j > 10)
      return false;
  }
  }

  return j > 0;
}

bool default_stmt0(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (b == true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
  }
  return false;
}

bool default_stmt1(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (b == false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(!b\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}
  }
  return false;
}

bool default_stmt2(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (b && true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
  }
  return false;
}

bool default_stmt3(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (b && false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(false\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}
  }
  return false;
}

bool default_stmt4(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (b || true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(true\)}}
    // CHECK-FIXES-NEXT: {{j = 10;}}
    // CHECK-FIXES-NEXT: {{break;}}
  }
  return false;
}

bool default_stmt5(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (b || false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{if \(b\)}}
    // CHECK-FIXES-NEXT: {{j = -20;}}
  }
  return false;
}

bool default_stmt6(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    return i > 0 ? true : false;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}} in ternary expression result
    // CHECK-FIXES: {{return i > 0;}}
  }
  return false;
}

bool default_stmt7(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    return i > 0 ? false : true;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: {{.*}} in ternary expression result
    // CHECK-FIXES: {{return i <= 0;}}
  }
  return false;
}

bool default_stmt8(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (true)
      j = 10;
    else
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in if statement condition
    // CHECK-FIXES:      {{j = 10;$}}
    // CHECK-FIXES-NEXT: {{break;$}}
  }
  return false;
}

bool default_stmt9(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (false)
      j = -20;
    else
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in if statement condition
    // CHECK-FIXES: {{j = 10;}}
    // CHECK-FIXES-NEXT: {{break;}}
  }
  return false;
}

bool default_stmt10(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      return true;
    else
      return false;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} in conditional return statement
    // CHECK-FIXES: {{return j > 10;}}
  }
  return false;
}

bool default_stmt11(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      return false;
    else
      return true;
    // CHECK-MESSAGES: :[[@LINE-3]]:14: warning: {{.*}} in conditional return statement
    // CHECK-FIXES: {{return j <= 10;}}
  }
  return false;
}

bool default_stmt12(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      b = true;
    else
      b = false;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: {{.*}} in conditional assignment
    // CHECK-FIXES: {{b = j > 10;}}
    // CHECK-FIXES-NEXT: {{return b;}}
  }
  return false;
}

bool default_stmt13(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      b = false;
    else
      b = true;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:11: warning: {{.*}} in conditional assignment
    // CHECK-FIXES: {{b = j <= 10;}}
    // CHECK-FIXES-NEXT: {{return b;}}
  }
  return false;
}

bool default_stmt14(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      return true;
    return false;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: {{.*}} in conditional return
    // FIXES: {{return j > 10;}}
  }
  return false;
}

bool default_stmt15(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      return false;
    return true;
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: {{.*}} in conditional return
    // FIXES: {{return j <= 10;}}
  }
  return false;
}

bool default_stmt16(int i, int j, bool b) {
  switch (i) {
  case 0:
    return false;

  default:
    if (j > 10)
      return true;
  }
  return false;
}

bool default_stmt17(int i, int j, bool b) {
  switch (i) {
  case 0:
    return true;

  default:
    if (j > 10)
      return false;
  }
  return false;
}

bool label_stmt0(int i, int j, bool b) {
label:
  if (b == true)
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} to boolean operator
  // CHECK-FIXES: {{if \(b\)}}
  // CHECK-FIXES-NEXT: {{j = 10;}}
  return false;
}

bool label_stmt1(int i, int j, bool b) {
label:
  if (b == false)
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} to boolean operator
  // CHECK-FIXES: {{if \(!b\)}}
  // CHECK-FIXES-NEXT: {{j = -20;}}
  return false;
}

bool label_stmt2(int i, int j, bool b) {
label:
  if (b && true)
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} to boolean operator
  // CHECK-FIXES: {{if \(b\)}}
  // CHECK-FIXES-NEXT: {{j = 10;}}
  return false;
}

bool label_stmt3(int i, int j, bool b) {
label:
  if (b && false)
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} to boolean operator
  // CHECK-FIXES: {{if \(false\)}}
  // CHECK-FIXES-NEXT: {{j = -20;}}
  return false;
}

bool label_stmt4(int i, int j, bool b) {
label:
  if (b || true)
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} to boolean operator
  // CHECK-FIXES: {{if \(true\)}}
  // CHECK-FIXES-NEXT: {{j = 10;}}
  return false;
}

bool label_stmt5(int i, int j, bool b) {
label:
  if (b || false)
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} to boolean operator
  // CHECK-FIXES: {{if \(b\)}}
  // CHECK-FIXES-NEXT: {{j = -20;}}
  return false;
}

bool label_stmt6(int i, int j, bool b) {
label:
  return i > 0 ? true : false;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{return i > 0;}}
}

bool label_stmt7(int i, int j, bool b) {
label:
  return i > 0 ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{return i <= 0;}}
}

bool label_stmt8(int i, int j, bool b) {
label:
  if (true)
    j = 10;
  else
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: {{.*}} in if statement condition
  // CHECK-FIXES:      {{j = 10;$}}
  return false;
}

bool label_stmt9(int i, int j, bool b) {
label:
  if (false)
    j = -20;
  else
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: {{.*}} in if statement condition
  // CHECK-FIXES: {{j = 10;}}
  return false;
}

bool label_stmt10(int i, int j, bool b) {
label:
  if (j > 10)
    return true;
  else
    return false;
  // CHECK-MESSAGES: :[[@LINE-3]]:12: warning: {{.*}} in conditional return statement
  // CHECK-FIXES: {{return j > 10;}}
}

bool label_stmt11(int i, int j, bool b) {
label:
  if (j > 10)
    return false;
  else
    return true;
  // CHECK-MESSAGES: :[[@LINE-3]]:12: warning: {{.*}} in conditional return statement
  // CHECK-FIXES: {{return j <= 10;}}
}

bool label_stmt12(int i, int j, bool b) {
label:
  if (j > 10)
    b = true;
  else
    b = false;
  return b;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: {{b = j > 10;}}
  // CHECK-FIXES-NEXT: {{return b;}}
}

bool label_stmt13(int i, int j, bool b) {
label:
  if (j > 10)
    b = false;
  else
    b = true;
  return b;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: {{b = j <= 10;}}
  // CHECK-FIXES-NEXT: {{return b;}}
}

bool label_stmt14(int i, int j, bool b) {
label:
  if (j > 10)
    return true;
  return false;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} in conditional return
  // FIXES: {{return j > 10;}}
}

bool label_stmt15(int i, int j, bool b) {
label:
  if (j > 10)
    return false;
  return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: {{.*}} in conditional return
  // FIXES: {{return j <= 10;}}
}
