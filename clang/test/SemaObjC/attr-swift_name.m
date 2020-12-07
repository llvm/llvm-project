// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

#define SWIFT_NAME(name) __attribute__((__swift_name__(name)))
#define SWIFT_ASYNC_NAME(name) __attribute__((__swift_async_name__(name)))

@protocol P
@end

typedef int (^CallbackTy)(void);

@interface AsyncI<P>

- (void)doSomethingWithCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething()");
- (void)doSomethingX:(int)x withCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething(x:)");

// expected-warning@+1 {{too many parameters in the signature specified by the '__swift_async_name__' attribute (expected 1; got 2)}}
- (void)doSomethingY:(int)x withCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething(x:y:)");

// expected-warning@+1 {{too few parameters in the signature specified by the '__swift_async_name__' attribute (expected 1; got 0)}}
- (void)doSomethingZ:(int)x withCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething()");

// expected-warning@+1 {{'__swift_async_name__' attribute cannot be applied to a method with no parameters}}
- (void)doSomethingNone SWIFT_ASYNC_NAME("doSomething()");

// expected-error@+1 {{'__swift_async_name__' attribute takes one argument}}
- (void)brokenAttr __attribute__((__swift_async_name__("brokenAttr", 2)));

@end

void asyncFunc(CallbackTy callback) SWIFT_ASYNC_NAME("asyncFunc()");

// expected-warning@+1 {{'__swift_async_name__' attribute cannot be applied to a function with no parameters}}
void asyncNoParams(void) SWIFT_ASYNC_NAME("asyncNoParams()");

// expected-error@+1 {{'__swift_async_name__' attribute only applies to Objective-C methods and functions}}
SWIFT_ASYNC_NAME("NoAsync")
@protocol NoAsync @end
