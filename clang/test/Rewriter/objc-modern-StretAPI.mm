// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

extern "C" void *sel_registerName(const char *);
typedef unsigned long size_t;

union U {
  double d1;
  char filler[32];
};

struct S {
  char filler[128];
};

@interface I
- (struct S) Meth : (int) arg1 : (id) arg2;
- (struct S) Meth1;
- (union U) Meth2 : (double)d;
- (struct S) VAMeth : (int)anchor, ...;
@end

I* PI();

extern "C" {

struct S foo () {
  struct S s = [PI() Meth : 1 : (id)0];

  U u = [PI() Meth2 : 3.14];

  S s1 = [PI() VAMeth : 12, 13.4, 1000, "hello"];

  S s2 = [PI() VAMeth : 12];

  S s3 = [PI() VAMeth : 0, "hello", "there"];

  S s4 = [PI() VAMeth : 2, ^{}, &foo];

  return [PI() Meth1];
}

}

