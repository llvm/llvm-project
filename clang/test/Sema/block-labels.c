// RUN: %clang_cc1 %s -verify -fblocks -fsyntax-only

void xx();

int a() { 
  A:
  
  if (1) xx();
  return ^{
         A: return 1;
       }();
}
int b() { 
  A: return ^{int a; A:return 1;}();
}

int d() { 
  A: return ^{int a; A: a = ^{int a; A:return 1;}() + ^{int b; A:return 2;}(); return a; }();
}

int c() { 
  goto A;     // expected-error {{use of undeclared label 'A'}}
  return ^{
       A:
        return 1;
     }();
}
