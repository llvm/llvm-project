// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

extern "C" void *sel_registerName(const char *);
typedef unsigned long size_t;

typedef struct {
    unsigned long long x;
    unsigned long long y;
} myPoint;

typedef struct {
    unsigned long long x;
    unsigned long long y;
} allPoint;

@interface Obj
+ (myPoint)foo;
+ (myPoint)foo : (int)Arg1 : (double)fArg;
+ (allPoint)fee;
@end

@implementation Obj
+ (allPoint)fee {
    allPoint a;
    a.x = a.y = 3;
    
    return a;
}
+ (myPoint)foo {
    myPoint r;
    r.x = 1;
    r.y = 2;
    return r;
}

+ (myPoint)foo : (int)Arg1 : (double)fArg {
  myPoint r;
  return r;
}
@end

myPoint Ret_myPoint() {
  return [Obj foo];
}

allPoint Ret_allPoint() {
  return [Obj fee];
}

myPoint Ret_myPoint1(int i, double d) {
  return [Obj foo:i:d];
}

myPoint Ret_myPoint2() {
  return [Obj foo];
}
