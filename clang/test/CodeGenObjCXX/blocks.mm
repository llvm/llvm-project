// RUN: %clang_cc1 -x objective-c++ -fblocks -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 %s -verify -std=c++11 -emit-llvm -o %t
// rdar://8979379

@interface A
@end

@interface B : A
@end

void f(int (^bl)(B* b));
void takeBlock(void (^block)());
void useValues(...);

// Test1
void g() {
  f(^(A* a) { return 0; });
}

// Test2
void g1() {
  int (^bl)(B* b) = ^(A* a) { return 0; };
}

// Test3
@protocol NSObject;

void bar(id(^)(void));

void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock);
}

// Test4
struct S {
  S *(^a)() = ^{
    return this;
  };
};
S s;

// Test5
struct X {
  void f() {
    ^ {
      struct Nested { Nested *ptr = this; };
    } ();
  };
};

// Regression test for PR13314
class FooClass { };
void fun() {
  FooClass foovar;
  ^() {  // expected-warning {{expression result unused}}
    return foovar;
  };
}
void gun() {
  FooClass foovar;
  [=]() {  // expected-warning {{expression result unused}}
    return foovar;
  };
}

// PR24780
class CaptureThisAndAnotherPointer {
  void test(void *ptr) {
    takeBlock(^{ useValues(ptr, this); });
  }
};
