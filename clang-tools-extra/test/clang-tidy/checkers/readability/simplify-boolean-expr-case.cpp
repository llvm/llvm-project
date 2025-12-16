// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t

bool switch_stmt(int i, int j, bool b) {
  switch (i) {
  case 0:
    if (b == true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 0:
    // CHECK-FIXES-NEXT: if (b)
    // CHECK-FIXES-NEXT:   j = 10;
    // CHECK-FIXES-NEXT: break;

  case 1:
    if (b == false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 1:
    // CHECK-FIXES-NEXT: if (!b)
    // CHECK-FIXES-NEXT:   j = -20;
    // CHECK-FIXES-NEXT: break;

  case 2:
    if (b && true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 2:
    // CHECK-FIXES-NEXT: if (b)
    // CHECK-FIXES-NEXT:   j = 10;
    // CHECK-FIXES-NEXT: break;

  case 3:
    if (b && false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 3:
    // CHECK-FIXES-NEXT: if (false)
    // CHECK-FIXES-NEXT:   j = -20;
    // CHECK-FIXES-NEXT: break;

  case 4:
    if (b || true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 4:
    // CHECK-FIXES-NEXT: if (true)
    // CHECK-FIXES-NEXT:   j = 10;
    // CHECK-FIXES-NEXT: break;

  case 5:
    if (b || false)
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 5:
    // CHECK-FIXES-NEXT: if (b)
    // CHECK-FIXES-NEXT:   j = -20;
    // CHECK-FIXES-NEXT: break;

  case 6:
    return i > 0 ? true : false;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: redundant boolean literal in ternary expression result [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 6:
    // CHECK-FIXES-NEXT: return i > 0;

  case 7:
    return i > 0 ? false : true;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: redundant boolean literal in ternary expression result [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 7:
    // CHECK-FIXES-NEXT: return i <= 0;

  case 8:
    if (true)
      j = 10;
    else
      j = -20;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:{{[0-9]+}}: warning: redundant boolean literal in if statement condition [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 8:
    // CHECK-FIXES-NEXT: j = 10;;
    // CHECK-FIXES-NEXT: break;

  case 9:
    if (false)
      j = -20;
    else
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-5]]:{{[0-9]+}}: warning: redundant boolean literal in if statement condition [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 9:
    // CHECK-FIXES-NEXT: j = 10;;
    // CHECK-FIXES-NEXT: break;

  case 10:
    if (j > 10)
      return true;
    else
      return false;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 10:
    // CHECK-FIXES-NEXT: return j > 10;

  case 11:
    if (j > 10)
      return false;
    else
      return true;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 11:
    // CHECK-FIXES-NEXT: return j <= 10;

  case 12:
    if (j > 10)
      b = true;
    else
      b = false;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning: redundant boolean literal in conditional assignment [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 12:
    // CHECK-FIXES-NEXT: b = j > 10;
    // CHECK-FIXES-NEXT: return b;

  case 13:
    if (j > 10)
      b = false;
    else
      b = true;
    return b;
    // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning: redundant boolean literal in conditional assignment [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 13:
    // CHECK-FIXES-NEXT: b = j <= 10;
    // CHECK-FIXES-NEXT: return b;

  case 14:
    if (j > 10)
      return true;
    return false;
    // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 14:
    // CHECK-FIXES-NEXT: return j > 10;

  case 15:
    if (j > 10)
      return false;
    return true;
    // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
    // CHECK-FIXES:      case 15:
    // CHECK-FIXES-NEXT: return j <= 10;

  default:
    if (b == true)
      j = 10;
    break;
    // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
    // CHECK-FIXES:      default:
    // CHECK-FIXES-NEXT: if (b)
    // CHECK-FIXES-NEXT:   j = 10;
    // CHECK-FIXES-NEXT: break;
  }
}

bool label_stmt0(int i, int j, bool b) {
label:
  if (b == true)
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
  // CHECK-FIXES:      if (b)
  // CHECK-FIXES-NEXT: j = 10;
  //
  //
}

bool label_stmt1(int i, int j, bool b) {
label:
  if (b == false)
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
  // CHECK-FIXES:      if (!b)
  // CHECK-FIXES-NEXT: j = -20;
  //
  //
}

bool label_stmt2(int i, int j, bool b) {
label:
  if (b && true)
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
  // CHECK-FIXES:      if (b)
  // CHECK-FIXES-NEXT: j = 10;
  //
  //
}

bool label_stmt3(int i, int j, bool b) {
label:
  if (b && false)
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
  // CHECK-FIXES:      if (false)
  // CHECK-FIXES-NEXT: j = -20;
  //
  //
}

bool label_stmt4(int i, int j, bool b) {
label:
  if (b || true)
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
  // CHECK-FIXES:      if (true)
  // CHECK-FIXES-NEXT: j = 10;
  //
  //
}

bool label_stmt5(int i, int j, bool b) {
label:
  if (b || false)
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
  // CHECK-FIXES:      if (b)
  // CHECK-FIXES-NEXT: j = -20;
  //
  //
}

bool label_stmt6(int i, int j, bool b) {
label:
  return i > 0 ? true : false;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: redundant boolean literal in ternary expression result [readability-simplify-boolean-expr]
  // CHECK-FIXES: return i > 0;
  //
  //
}

bool label_stmt7(int i, int j, bool b) {
label:
  return i > 0 ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: redundant boolean literal in ternary expression result [readability-simplify-boolean-expr]
  // CHECK-FIXES: return i <= 0;
  //
  //
}

bool label_stmt8(int i, int j, bool b) {
label:
  if (true)
    j = 10;
  else
    j = -20;
  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning: redundant boolean literal in if statement condition [readability-simplify-boolean-expr]
  // CHECK-FIXES: j = 10;;
  //
  //
  return false;
}

bool label_stmt9(int i, int j, bool b) {
label:
  if (false)
    j = -20;
  else
    j = 10;
  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning: redundant boolean literal in if statement condition [readability-simplify-boolean-expr]
  // CHECK-FIXES: j = 10;;
  //
  //
  return false;
}

bool label_stmt10(int i, int j, bool b) {
label:
  if (j > 10)
    return true;
  else
    return false;
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
  // CHECK-FIXES: return j > 10;
  //
  //
}

bool label_stmt11(int i, int j, bool b) {
label:
  if (j > 10)
    return false;
  else
    return true;
  // CHECK-MESSAGES: :[[@LINE-3]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
  // CHECK-FIXES: return j <= 10;
  //
  //
}

bool label_stmt12(int i, int j, bool b) {
label:
  if (j > 10)
    b = true;
  else
    b = false;
  return b;
  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning: redundant boolean literal in conditional assignment [readability-simplify-boolean-expr]
  // CHECK-FIXES: b = j > 10;
  // CHECK-FIXES-NEXT: return b;
  //
  //
}

bool label_stmt13(int i, int j, bool b) {
label:
  if (j > 10)
    b = false;
  else
    b = true;
  return b;
  // CHECK-MESSAGES: :[[@LINE-4]]:{{[0-9]+}}: warning: redundant boolean literal in conditional assignment [readability-simplify-boolean-expr]
  // CHECK-FIXES: b = j <= 10;
  // CHECK-FIXES-NEXT: return b;
  //
  //
}

bool label_stmt14(int i, int j, bool b) {
label:
  if (j > 10)
    return true;
  return false;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
  // CHECK-FIXES: return j > 10;
}

bool label_stmt15(int i, int j, bool b) {
label:
  if (j > 10)
    return false;
  return true;
  // CHECK-MESSAGES: :[[@LINE-2]]:{{[0-9]+}}: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
  // CHECK-FIXES: return j <= 10;
}
