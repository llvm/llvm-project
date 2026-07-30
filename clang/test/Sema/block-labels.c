// RUN: %clang_cc1 %s -verify -fblocks -fsyntax-only

void xx(void);

int a(void) { 
  A:
  
  if (1) xx();
  return ^{
         A: return 1;
       }();
}
int b(void) { 
  A: return ^{int a; A:return 1;}();
}

int d(void) { 
  A: return ^{int a; A: a = ^{int a; A:return 1;}() + ^{int b; A:return 2;}(); return a; }();
}

int c(void) { 
  goto A;     // expected-error {{use of undeclared label 'A'}}
  return ^{
       A:
        return 1;
     }();
}
