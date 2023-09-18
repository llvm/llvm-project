// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@class NSData, NSError;

@interface Foo

typedef void (^Callback)(NSData *data, NSError *error);

- (void)doSomething:(NSData *)data callback:(Callback)callback;
@end

@implementation Foo

- (void)doSomething:(NSData *)data callback:(Callback)callback {
  callback(0, 0);
}

@end
