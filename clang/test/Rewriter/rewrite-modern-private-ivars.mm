// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %s -o %t-rw.cpp
// RUN: %clang_cc1 -Werror -fsyntax-only -Wno-address-of-temporary -Wno-c++11-narrowing -std=c++11 -D"Class=void*" -D"id=void*" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp
// rdar://11351299

struct Q {
  int x;
};

@interface I
@end

@interface I() {

  struct {
    int x;
  } unnamed;

  struct S {
    int x;
  } foo;

  double dd;

  struct S foo1;
}
@end

@implementation I 
{
  struct P {
    int x;
  } bar;

  double ee;

  struct Q bar1;

  struct {
    int x;
  } noname;
}

- (void) Meth { 
  foo.x = 1; 
  bar.x = 2; 
  dd = 1.23; 
  ee = 0.0; 
  foo1.x = 3;
  bar1.x = 4;
  noname.x = 3;
  unnamed.x = 10;
}
@end
