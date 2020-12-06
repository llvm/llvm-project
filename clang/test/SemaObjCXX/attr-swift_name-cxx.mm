// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

#define SWIFT_ASYNC_NAME(name) __attribute__((__swift_async_name__(name)))

typedef int (^CallbackTy)(void);

class CXXClass {
public:
  virtual void doSomethingWithCallback(CallbackTy callback) SWIFT_ASYNC_NAME("doSomething()");

  // expected-warning@+1 {{too few parameters in the signature specified by the '__swift_async_name__' attribute (expected 1; got 0)}}
  virtual void doSomethingWithCallback(int x, CallbackTy callback) SWIFT_ASYNC_NAME("doSomething()");
};
