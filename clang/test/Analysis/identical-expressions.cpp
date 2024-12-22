// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core.IdenticalExpr -w -verify %s

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
  return ((int)f != (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkNotEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F != f + 3.14F);  // no warning
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
  return (0);
}




/* end '!=' with float*/

/* '!=' with int*/

int checkNotEqualIntLiteralCompare1(void) {
  return (5 != 5); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return ((int)f != (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (f + 4 != f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = ((int)f + 4 != (int)f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (u + t != u + t);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) != ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*(t + 1 != t + 1)));  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return ((int*)f != (int*)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (f + 4 != f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = ((int*)f + 4 != (int*)f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = ((f + (3)*t) != (f + (3)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return (f+1 == f+1); // expected-warning {{comparison of identical expressions always evaluates to true}}
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
  return ((int)f == (int)f); // expected-warning {{comparison of identical expressions always evaluates to true}}
}
int checkEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f == (int)f); // no warning
}
int checkEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F == f + 3.14F);  // no warning
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
  res = ((int)f + 3.14F == (int)f + 3.14F);  // no warning
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
  res = (((int)f + (3.14F - u)*t) == ((int)f + (3.14F - u)*t));  // no warning
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
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*(f + t == f + t)));  // no warning
  return (0);
}





/* Equal with int*/

int checkEqualIntLiteralCompare1(void) {
  return (5 == 5); // expected-warning {{comparison of identical expressions always evaluates to true}}
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
  return ((int)f == (int)f); // expected-warning {{comparison of identical expressions always evaluates to true}}
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
  res = (f + 4 == f + 4);  // expected-warning {{comparison of identical expressions always evaluates to true}}
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
  res = ((int)f + 4 == (int)f + 4);  // expected-warning {{comparison of identical expressions always evaluates to true}}
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
  res = (u + t == u + t);  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);
}

int checkEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) == ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to true}}
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
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*(t + 1 == t + 1)));  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);
}

/* '==' with function*/

int checkEqualSameFunction() {
  unsigned a = 0;
  unsigned b = 1;
  int res = (a+func() == a+func());  // no warning
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
  int res = (a+funcParam(a) == a+funcParam(a));  // no warning
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
  return (5.14F < 5.14F); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return (f < f); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return ((int)f < (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkLessThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F < f + 3.14F);  // no warning
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
  res = ((int)f + 3.14F < (int)f + 3.14F);  // no warning
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
  res = (((int)f + (3.14F - u)*t) < ((int)f + (3.14F - u)*t));  // no warning
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
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*(f + t < f + t)));  // no warning
  return (0);
}

/*  end LT with float */

/*  LT with int */


int checkLessThanIntLiteralCompare1(void) {
  return (5 < 5); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return ((int)f < (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkLessThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 < f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = ((int)f + 3 < (int)f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (((int)f + (3 - u)*t) < ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*(t + u < t + u)));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

/* end LT with int */

/* end LT */


/* GT */

/* GT with float */

int checkGreaterThanFloatLiteralCompare1(void) {
  return (5.14F > 5.14F); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return (f > f); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return ((int)f > (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkGreaterThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F > f + 3.14F);  // no warning
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
  res = ((int)f + 3.14F > (int)f + 3.14F);  // no warning
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
  res = (((int)f + (3.14F - u)*t) > ((int)f + (3.14F - u)*t));  // no warning
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
  return (0);
}

/*  end GT with float */

/*  GT with int */


int checkGreaterThanIntLiteralCompare1(void) {
  return (5 > 5); // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  return ((int)f > (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkGreaterThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 > f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = ((int)f + 3 > (int)f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (((int)f + (3 - u)*t) > ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
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
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*(t + u > t + u)));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

/* end GT with int */

/* end GT */


/* Checking use of identical expressions in conditional operator*/

unsigned test_unsigned(unsigned a) {
  unsigned b = 1;
  a = a > 5 ? b : b; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
  return a;
}

void test_signed() {
  int a = 0;
  a = a > 5 ? a : a; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_bool(bool a) {
  a = a > 0 ? a : a; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_float() {
  float a = 0;
  float b = 0;
  a = a > 5 ? a : a; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

const char *test_string() {
  float a = 0;
  return a > 5 ? "abc" : "abc"; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_unsigned_expr() {
  unsigned a = 0;
  unsigned b = 0;
  a = a > 5 ? a+b : a+b; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_signed_expr() {
  int a = 0;
  int b = 1;
  a = a > 5 ? a+b : a+b; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_bool_expr(bool a) {
  bool b = 0;
  a = a > 0 ? a&&b : a&&b; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
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
  a = a > 5 ? a+b : a+b; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_expr_positive_func() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+func() : a+func(); // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_expr_negative_func() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+func() : a+func2(); // no warning
}

void test_expr_positive_funcParam() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+funcParam(b) : a+funcParam(b); // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_expr_negative_funcParam() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a+funcParam(a) : a+funcParam(b); // no warning
}

void test_expr_positive_inc() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a++ : a++; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_expr_negative_inc() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a++ : b++; // no warning
}

void test_expr_positive_assign() {
  unsigned a = 0;
  unsigned b = 1;
  a = a > 5 ? a=1 : a=1;  // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
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
  a = a > 5 ? a+b+(c+a)*(a + b*(c+a)) : a+b+(c+a)*(a + b*(c+a)); // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
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
  a = a > 5 ? (b > 5 ? 1 : 4) : (b > 5 ? 4 : 4); // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}

void test_identical_branches1(bool b) {
  int i = 0;
  if (b) { // expected-warning {{true and false branches are identical}}
    ++i;
  } else {
    ++i;
  }
}

void test_identical_branches2(bool b) {
  int i = 0;
  if (b) { // expected-warning {{true and false branches are identical}}
    ++i;
  } else
    ++i;
}

void test_identical_branches3(bool b) {
  int i = 0;
  if (b) { // no warning
    ++i;
  } else {
    i++;
  }
}

void test_identical_branches4(bool b) {
  int i = 0;
  if (b) { // expected-warning {{true and false branches are identical}}
  } else {
  }
}

void test_identical_branches_break(bool b) {
  while (true) {
    if (b) // expected-warning {{true and false branches are identical}}
      break;
    else
      break;
  }
}

void test_identical_branches_continue(bool b) {
  while (true) {
    if (b) // expected-warning {{true and false branches are identical}}
      continue;
    else
      continue;
  }
}

void test_identical_branches_func(bool b) {
  if (b) // expected-warning {{true and false branches are identical}}
    func();
  else
    func();
}

void test_identical_branches_func_arguments(bool b) {
  if (b) // no-warning
    funcParam(1);
  else
    funcParam(2);
}

void test_identical_branches_cast1(bool b) {
  long v = -7;
  if (b) // no-warning
    v = (signed int) v;
  else
    v = (unsigned int) v;
}

void test_identical_branches_cast2(bool b) {
  long v = -7;
  if (b) // expected-warning {{true and false branches are identical}}
    v = (signed int) v;
  else
    v = (signed int) v;
}

int test_identical_branches_return_int(bool b) {
  int i = 0;
  if (b) { // expected-warning {{true and false branches are identical}}
    i++;
    return i;
  } else {
    i++;
    return i;
  }
}

int test_identical_branches_return_func(bool b) {
  if (b) { // expected-warning {{true and false branches are identical}}
    return func();
  } else {
    return func();
  }
}

void test_identical_branches_for(bool b) {
  int i;
  int j;
  if (b) { // expected-warning {{true and false branches are identical}}
    for (i = 0, j = 0; i < 10; i++)
      j += 4;
  } else {
    for (i = 0, j = 0; i < 10; i++)
      j += 4;
  }
}

void test_identical_branches_while(bool b) {
  int i = 10;
  if (b) { // expected-warning {{true and false branches are identical}}
    while (func())
      i--;
  } else {
    while (func())
      i--;
  }
}

void test_identical_branches_while_2(bool b) {
  int i = 10;
  if (b) { // no-warning
    while (func())
      i--;
  } else {
    while (func())
      i++;
  }
}

void test_identical_branches_do_while(bool b) {
  int i = 10;
  if (b) { // expected-warning {{true and false branches are identical}}
    do {
      i--;
    } while (func());
  } else {
    do {
      i--;
    } while (func());
  }
}

void test_identical_branches_if(bool b, int i) {
  if (b) { // expected-warning {{true and false branches are identical}}
    if (i < 5)
      i += 10;
  } else {
    if (i < 5)
      i += 10;
  }
}

void test_identical_bitwise1() {
  int a = 5 | 5; // expected-warning {{identical expressions on both sides of bitwise operator}}
}

void test_identical_bitwise2() {
  int a = 5;
  int b = a | a; // expected-warning {{identical expressions on both sides of bitwise operator}}
}

void test_identical_bitwise3() {
  int a = 5;
  int b = (a | a); // expected-warning {{identical expressions on both sides of bitwise operator}}
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
  int b = a | 4 | a; // expected-warning {{identical expressions on both sides of bitwise operator}}
}

void test_identical_bitwise7() {
  int a = 5;
  int b = func() | func(); // no-warning
}

void test_identical_logical1(int a) {
  if (a == 4 && a == 4) // expected-warning {{identical expressions on both sides of logical operator}}
    ;
}

void test_identical_logical2(int a) {
  if (a == 4 || a == 5 || a == 4) // expected-warning {{identical expressions on both sides of logical operator}}
    ;
}

void test_identical_logical3(int a) {
  if (a == 4 || a == 5 || a == 6) // no-warning
    ;
}

void test_identical_logical4(int a) {
  if (a == func() || a == func()) // no-warning
    ;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlogical-op-parentheses"
void test_identical_logical5(int x, int y) {
  if (x == 4 && y == 5 || x == 4 && y == 6) // no-warning
    ;
}

void test_identical_logical6(int x, int y) {
  if (x == 4 && y == 5 || x == 4 && y == 5) // expected-warning {{identical expressions on both sides of logical operator}}
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

void test_warn_chained_if_stmts_1(int x) {
  if (x == 1)
    ;
  else if (x == 1) // expected-warning {{expression is identical to previous condition}}
    ;
}

void test_warn_chained_if_stmts_2(int x) {
  if (x == 1)
    ;
  else if (x == 1) // expected-warning {{expression is identical to previous condition}}
    ;
  else if (x == 1) // expected-warning {{expression is identical to previous condition}}
    ;
}

void test_warn_chained_if_stmts_3(int x) {
  if (x == 1)
    ;
  else if (x == 2)
    ;
  else if (x == 1) // expected-warning {{expression is identical to previous condition}}
    ;
}

void test_warn_chained_if_stmts_4(int x) {
  if (x == 1)
    ;
  else if (func())
    ;
  else if (x == 1) // expected-warning {{expression is identical to previous condition}}
    ;
}

void test_warn_chained_if_stmts_5(int x) {
  if (x & 1)
    ;
  else if (x & 1) // expected-warning {{expression is identical to previous condition}}
    ;
}

void test_warn_chained_if_stmts_6(int x) {
  if (x == 1)
    ;
  else if (x == 2)
    ;
  else if (x == 2) // expected-warning {{expression is identical to previous condition}}
    ;
  else if (x == 3)
    ;
}

void test_warn_chained_if_stmts_7(int x) {
  if (x == 1)
    ;
  else if (x == 2)
    ;
  else if (x == 3)
    ;
  else if (x == 2) // expected-warning {{expression is identical to previous condition}}
    ;
  else if (x == 5)
    ;
}

void test_warn_chained_if_stmts_8(int x) {
  if (x == 1)
    ;
  else if (x == 2)
    ;
  else if (x == 3)
    ;
  else if (x == 2) // expected-warning {{expression is identical to previous condition}}
    ;
  else if (x == 5)
    ;
  else if (x == 3) // expected-warning {{expression is identical to previous condition}}
    ;
  else if (x == 7)
    ;
}

void test_nowarn_chained_if_stmts_1(int x) {
  if (func())
    ;
  else if (func()) // no-warning
    ;
}

void test_nowarn_chained_if_stmts_2(int x) {
  if (func())
    ;
  else if (x == 1)
    ;
  else if (func()) // no-warning
    ;
}

void test_nowarn_chained_if_stmts_3(int x) {
  if (x++)
    ;
  else if (x++) // no-warning
    ;
}

void test_warn_wchar() {
  const wchar_t * a = 0 ? L"Warning" : L"Warning"; // expected-warning {{identical expressions on both sides of ':' in conditional expression}}
}
void test_nowarn_wchar() {
  const wchar_t * a = 0 ? L"No" : L"Warning";
}

void test_nowarn_long() {
  int a = 0, b = 0;
  long c;
  if (0) {
    b -= a;
    c = 0;
  } else { // no-warning
    b -= a;
    c = 0LL;
  }
}

// Identical inner conditions

void test_warn_inner_if_1(int x) {
  if (x == 1) {
    if (x == 1) // expected-warning {{conditions of the inner and outer statements are identical}}
      ;
  }

  // FIXME: Should warn here. The warning is currently not emitted because there
  // is code between the conditions.
  if (x == 1) {
    int y = x;
    if (x == 1)
      ;
  }
}

void test_nowarn_inner_if_1(int x) {
  // Don't warn when condition has side effects.
  if (x++ == 1) {
    if (x++ == 1)
      ;
  }

  // Don't warn when x is changed before inner condition.
  if (x < 10) {
    x++;
    if (x < 10)
      ;
  }
}
