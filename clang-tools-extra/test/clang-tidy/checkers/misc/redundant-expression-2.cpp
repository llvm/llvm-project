// RUN: %check_clang_tidy %s misc-redundant-expression -check-suffix=IDENTEXPR %t

/* Only one expected warning per function allowed at the very end. */

int func(void)
{
  return 0;
}

int func2(void)
{
  return 0;
}

int funcParam(int a)
{
  return 0;
}

/* '!=' operator*/

/* '!=' with float */
int checkNotEqualFloatLiteralCompare1(void) {
  return (5.14F != 5.14F); // no warning
}

int checkNotEqualFloatLiteralCompare2(void) {
  return (6.14F != 7.14F); // no warning
}

int checkNotEqualFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f != g); // no warning
}

int checkNotEqualFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f != f); // no warning
}

int checkNotEqualFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f != 7.1F); // no warning
}

int checkNotEqualFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F != f); // no warning
}

int checkNotEqualFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t != f); // no warning
}

int checkNotEqualFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f != t); // no warning
}



int checkNotEqualCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f != (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkNotEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F != f + 3.14F);  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:20: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F != g + 3.14F); // no warning
}
int checkNotEqualBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F != (int)f + 3.14F);  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:25: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F != (char)f + 3.14F);  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) != ((int)f + (3.14F - u)*t));  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:35: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) != ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) != ((int)f + (3.14F - u)*(f + t != f + t)));  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:67: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}


/* end '!=' with float*/

/* '!=' with int*/

int checkNotEqualIntLiteralCompare1(void) {
  return (5 != 5);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkNotEqualIntLiteralCompare2(void) {
  return (6 != 7); // no warning
}

int checkNotEqualIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f != g); // no warning
}

int checkNotEqualIntDeclCompare3(void) {
  int f = 7;
  return (f != 7); // no warning
}

int checkNotEqualIntDeclCompare4(void) {
  int f = 7;
  return (7 != f); // no warning
}

int checkNotEqualCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f != (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkNotEqualCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = (f + 4 != f + 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 4 != g + 4); // no warning
}


int checkNotEqualBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 != (int)f + 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntCompare4(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 != (char)f + 4);  // no warning
  return (0);
}
int checkNotEqualBinaryOpIntCompare5(void) {
  int res;
  int t= 1;
  int u= 2;
  res = (u + t != u + t);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) != ((int)f + (3 - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*(t + 1 != t + 1)));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:59: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/*   end '!=' int          */

/* '!=' with int pointer */

int checkNotEqualIntPointerLiteralCompare1(void) {
  int* p = 0;
  return (p != 0); // no warning
}

int checkNotEqualIntPointerLiteralCompare2(void) {
  return (6 != 7); // no warning
}

int checkNotEqualIntPointerDeclCompare1(void) {
  int k = 3;
  int* f = &k;
  int* g = &k;
  return (f != g); // no warning
}

int checkNotEqualCastIntPointerDeclCompare11(void) {
  int k = 7;
  int* f = &k;
  return ((int*)f != (int*)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:19: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkNotEqualCastIntPointerDeclCompare12(void) {
  int k = 7;
  int* f = &k;
  return ((int*)((char*)f) != (int*)f); // no warning
}
int checkNotEqualBinaryOpIntPointerCompare1(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = (f + 4 != f + 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntPointerCompare2(void) {
  int k = 7;
  int* f = &k;
  int* g = &k;
  return (f + 4 != g + 4); // no warning
}


int checkNotEqualBinaryOpIntPointerCompare3(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = ((int*)f + 4 != (int*)f + 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:22: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkNotEqualBinaryOpIntPointerCompare4(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = ((int*)f + 4 != (int*)((char*)f) + 4);  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpIntPointerCompare1(void) {
  int res;
  int k = 7;
  int t= 1;
  int* u= &k+2;
  int* f= &k+3;
  res = ((f + (3)*t) != (f + (3)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:22: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualNestedBinaryOpIntPointerCompare2(void) {
  int res;
  int k = 7;
  int t= 1;
  int* u= &k+2;
  int* f= &k+3;
  res = (((3)*t + f) != (f + (3)*t));  // no warning
  return (0);
}
/*   end '!=' int*          */

/* '!=' with function*/

int checkNotEqualSameFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() != a+func());  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:23: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualDifferentFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() != a+func2());  // no warning
  return (0);
}

int checkNotEqualSameFunctionSameParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) != a+funcParam(a));  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:29: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkNotEqualSameFunctionDifferentParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) != a+funcParam(b));  // no warning
  return (0);
}

/*   end '!=' with function*/

/*   end '!=' */



/* EQ operator           */

int checkEqualIntPointerDeclCompare(void) {
  int k = 3;
  int* f = &k;
  int* g = &k;
  return (f == g); // no warning
}

int checkEqualIntPointerDeclCompare0(void) {
  int k = 3;
  int* f = &k;
  return (f+1 == f+1);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:15: warning: both sides of operator are equivalent [misc-redundant-expression]
}

/* EQ with float*/

int checkEqualFloatLiteralCompare1(void) {
  return (5.14F == 5.14F); // no warning
}

int checkEqualFloatLiteralCompare2(void) {
  return (6.14F == 7.14F); // no warning
}

int checkEqualFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f == g); // no warning
}

int checkEqualFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f == f); // no warning
}


int checkEqualFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f == 7.1F); // no warning
}

int checkEqualFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F == f); // no warning
}

int checkEqualFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t == f); // no warning
}

int checkEqualFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f == t); // no warning
}




int checkEqualCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f == (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f == (int)f); // no warning
}
int checkEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F == f + 3.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:20: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkEqualBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F == g + 3.14F); // no warning
}
int checkEqualBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F == (int)f + 3.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:25: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkEqualBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F == (char)f + 3.14F);  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) == ((int)f + (3.14F - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:35: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*(f + t == f + t)));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:67: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* Equal with int*/

int checkEqualIntLiteralCompare1(void) {
  return (5 == 5);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkEqualIntLiteralCompare2(void) {
  return (6 == 7); // no warning
}

int checkEqualIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f == g); // no warning
}

int checkEqualCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f == (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkEqualCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f == (int)f); // no warning
}

int checkEqualIntDeclCompare3(void) {
  int f = 7;
  return (f == 7); // no warning
}

int checkEqualIntDeclCompare4(void) {
  int f = 7;
  return (7 == f); // no warning
}

int checkEqualBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = (f + 4 == f + 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkEqualBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 4 == g + 4); // no warning
}


int checkEqualBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 == (int)f + 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);

}
int checkEqualBinaryOpIntCompare4(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 == (char)f + 4);  // no warning
  return (0);
}
int checkEqualBinaryOpIntCompare5(void) {
  int res;
  int t= 1;
  int u= 2;
  res = (u + t == u + t);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) == ((int)f + (3 - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*(t + 1 == t + 1)));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:59: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* '==' with function*/

int checkEqualSameFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() == a+func());
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:23: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualDifferentFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() == a+func2());  // no warning
  return (0);
}

int checkEqualSameFunctionSameParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) == a+funcParam(a));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:29: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkEqualSameFunctionDifferentParam() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+funcParam(a) == a+funcParam(b));  // no warning
  return (0);
}

/*   end '==' with function*/

/*   end EQ int          */

/* end EQ */


/*  LT */

/*  LT with float */

int checkLessThanFloatLiteralCompare1(void) {
  return (5.14F < 5.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanFloatLiteralCompare2(void) {
  return (6.14F < 7.14F); // no warning
}

int checkLessThanFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f < g); // no warning
}

int checkLessThanFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f < f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f < 7.1F); // no warning
}

int checkLessThanFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F < f); // no warning
}

int checkLessThanFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t < f); // no warning
}

int checkLessThanFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f < t); // no warning
}


int checkLessThanCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f < (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkLessThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F < f + 3.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:20: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkLessThanBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F < g + 3.14F); // no warning
}
int checkLessThanBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F < (int)f + 3.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:25: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkLessThanBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F < (char)f + 3.14F);  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) < ((int)f + (3.14F - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:35: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*(f + t < f + t)));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:66: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/*  end LT with float */

/*  LT with int */


int checkLessThanIntLiteralCompare1(void) {
  return (5 < 5);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkLessThanIntLiteralCompare2(void) {
  return (6 < 7); // no warning
}

int checkLessThanIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f < g); // no warning
}

int checkLessThanIntDeclCompare3(void) {
  int f = 7;
  return (f < 7); // no warning
}

int checkLessThanIntDeclCompare4(void) {
  int f = 7;
  return (7 < f); // no warning
}

int checkLessThanIntDeclCompare5(void) {
  int f = 7;
  int t = 7;
  return (t < f); // no warning
}

int checkLessThanIntDeclCompare6(void) {
  int f = 7;
  int t = 7;
  return (f < t); // no warning
}

int checkLessThanCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f < (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkLessThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 < f + 3);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkLessThanBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 3 < g + 3); // no warning
}
int checkLessThanBinaryOpIntCompare3(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 < (int)f + 3);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkLessThanBinaryOpIntCompare4(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 < (char)f + 3);  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) < ((int)f + (3 - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*(t + u < t + u)));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:58: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* end LT with int */

/* end LT */


/* GT */

/* GT with float */

int checkGreaterThanFloatLiteralCompare1(void) {
  return (5.14F > 5.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkGreaterThanFloatLiteralCompare2(void) {
  return (6.14F > 7.14F); // no warning
}

int checkGreaterThanFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;

  return (f > g); // no warning
}

int checkGreaterThanFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f > f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}


int checkGreaterThanFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f > 7.1F); // no warning
}

int checkGreaterThanFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F > f); // no warning
}

int checkGreaterThanFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t > f); // no warning
}

int checkGreaterThanFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f > t); // no warning
}

int checkGreaterThanCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f > (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkGreaterThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F > f + 3.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:20: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkGreaterThanBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F > g + 3.14F); // no warning
}
int checkGreaterThanBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F > (int)f + 3.14F);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:25: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkGreaterThanBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F > (char)f + 3.14F);  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) > ((int)f + (3.14F - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:35: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) > ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) > ((int)f + (3.14F - u)*(f + t > f + t)));  // no warning
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:66: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/*  end GT with float */

/*  GT with int */


int checkGreaterThanIntLiteralCompare1(void) {
  return (5 > 5);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

int checkGreaterThanIntLiteralCompare2(void) {
  return (6 > 7); // no warning
}

int checkGreaterThanIntDeclCompare1(void) {
  int f = 7;
  int g = 7;

  return (f > g); // no warning
}

int checkGreaterThanIntDeclCompare3(void) {
  int f = 7;
  return (f > 7); // no warning
}

int checkGreaterThanIntDeclCompare4(void) {
  int f = 7;
  return (7 > f); // no warning
}

int checkGreaterThanCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f > (int)f);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}
int checkGreaterThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 > f + 3);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:16: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkGreaterThanBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 3 > g + 3); // no warning
}
int checkGreaterThanBinaryOpIntCompare3(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 > (int)f + 3);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:21: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}
int checkGreaterThanBinaryOpIntCompare4(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 > (char)f + 3);  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) > ((int)f + (3 - u)*t));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:31: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*(t + u > t + u)));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:58: warning: both sides of operator are equivalent [misc-redundant-expression]
  return (0);
}

/* end GT with int */

/* end GT */


/* Checking use of identical expressions in conditional operator*/

unsigned test_unsigned(unsigned a) {
  unsigned b = 1;
  a = a > 5 ? b : b;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
  return a;
}

void test_signed() {
  int a = 0;
  a = a > 5 ? a : a;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_bool(bool a) {
  a = a > 0 ? a : a;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_float() {
  float a = 0;
  float b = 0;
  a = a > 5 ? a : a;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

const char *test_string() {
  float a = 0;
  return a > 5 ? "abc" : "abc";
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:24: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_unsigned_expr() {
  unsigned a = 0;
  unsigned b = 0;
  a = a > 5 ? a+b : a+b;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:19: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_signed_expr() {
  int a = 0;
  int b = 1;
  a = a > 5 ? a+b : a+b;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:19: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_bool_expr(bool a) {
  bool b = 0;
  a = a > 0 ? a&&b : a&&b;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:20: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_unsigned_expr_negative() {
  unsigned a = 0;
  unsigned b = 0;
  a = a > 5 ? a+b : b+a; // no warning
}

void test_signed_expr_negative() {
  int a = 0;
  int b = 1;
  a = a > 5 ? b+a : a+b; // no warning
}

void test_bool_expr_negative(bool a) {
  bool b = 0;
  a = a > 0 ? a&&b : b&&a; // no warning
}

void test_float_expr_positive() {
  float a = 0;
  float b = 0;
  a = a > 5 ? a+b : a+b;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:19: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_expr_positive_func() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+func() : a+func();
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:24: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_expr_negative_func() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+func() : a+func2(); // no warning
}

void test_expr_positive_funcParam() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+funcParam(b) : a+funcParam(b);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:30: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_expr_negative_funcParam() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+funcParam(a) : a+funcParam(b); // no warning
}

void test_expr_negative_inc() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a++ : b++; // no warning
}

void test_expr_negative_assign() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a=1 : a=2; // no warning
}

void test_signed_nested_expr() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? a+b+(c+a)*(a + b*(c+a)) : a+b+(c+a)*(a + b*(c+a));
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:39: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_signed_nested_expr_negative() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? a+b+(c+a)*(a + b*(c+a)) : a+b+(c+a)*(a + b*(a+c)); // no warning
}

void test_signed_nested_cond_expr_negative() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? (b > 5 ? 1 : 4) : (b > 5 ? 2 : 4); // no warning
}

void test_signed_nested_cond_expr() {
  int a = 0;
  int b = 1;
  int c = 3;
  a = a > 5 ? (b > 5 ? 1 : 4) : (b > 5 ? 4 : 4);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:44: warning: 'true' and 'false' expressions are equivalent [misc-redundant-expression]
}

void test_identical_bitwise1() {
  int a = 5 | 5;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

void test_identical_bitwise2() {
  int a = 5;
  int b = a | a;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:13: warning: both sides of operator are equivalent [misc-redundant-expression]
}

void test_identical_bitwise3() {
  int a = 5;
  int b = (a | a);
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:14: warning: both sides of operator are equivalent [misc-redundant-expression]
}

void test_identical_bitwise4() {
  int a = 4;
  int b = a | 4; // no-warning
}

void test_identical_bitwise5() {
  int a = 4;
  int b = 4;
  int c = a | b; // no-warning
}

void test_identical_bitwise6() {
  int a = 5;
  int b = a | 4 | a;
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:17: warning: operator has equivalent nested operands [misc-redundant-expression]
}

void test_identical_bitwise7() {
  int a = 5;
  int b = func() | func();
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:18: warning: both sides of operator are equivalent [misc-redundant-expression]
}

void test_identical_logical1(int a) {
  if (a == 4 && a == 4)
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:14: warning: both sides of operator are equivalent [misc-redundant-expression]
    ;
}

void test_identical_logical2(int a) {
  if (a == 4 || a == 5 || a == 4)
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:24: warning: operator has equivalent nested operands [misc-redundant-expression]
    ;
}

void test_identical_logical3(int a) {
  if (a == 4 || a == 5 || a == 6) // no-warning
    ;
}

void test_identical_logical4(int a) {
  if (a == func() || a == func())
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:19: warning: both sides of operator are equivalent [misc-redundant-expression]
    ;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlogical-op-parentheses"
void test_identical_logical5(int x, int y) {
  if (x == 4 && y == 5 || x == 4 && y == 6) // no-warning
    ;
}

void test_identical_logical6(int x, int y) {
  if (x == 4 && y == 5 || x == 4 && y == 5)
// CHECK-MESSAGES-IDENTEXPR: :[[@LINE-1]]:24: warning: both sides of operator are equivalent [misc-redundant-expression]
    ;
}

void test_identical_logical7(int x, int y) {
  // FIXME: We should warn here
  if (x == 4 && y == 5 || x == 4)
    ;
}

void test_identical_logical8(int x, int y) {
  // FIXME: We should warn here
  if (x == 4 || y == 5 && x == 4)
    ;
}

void test_identical_logical9(int x, int y) {
  // FIXME: We should warn here
  if (x == 4 || x == 4 && y == 5)
    ;
}
#pragma clang diagnostic pop
