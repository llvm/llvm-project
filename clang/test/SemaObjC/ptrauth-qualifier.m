// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -fptrauth-intrinsics %s

#if !__has_extension(ptrauth_qualifier)
// This error means that the __ptrauth qualifier availability test says  that it
// is not available. This error is not expected in the output, if it is seen
// there is a feature detection regression.
#error __ptrauth qualifier not enabled
#endif

@interface Foo
// expected-warning@-1 {{class 'Foo' defined without specifying a base class}}
// expected-note@-2 {{add a super class to fix this problem}}

@property void *__ptrauth(1, 1, 1) invalid1;
// expected-error@-1 {{property may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,1,1)'}}

@property void *__ptrauth(1, 0, 1) invalid2;
// expected-error@-1 {{property may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,0,1)'}}

- (void *__ptrauth(1, 1, 1))invalid5;
// expected-error@-1 {{return type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,1,1)'}}

- (void *__ptrauth(1, 0, 1))invalid6;
// expected-error@-1 {{return type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,0,1)'}}

- (void)invalid9:(void *__ptrauth(1, 1, 1))a;
// expected-error@-1 {{parameter type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,1,1)'}}
// expected-note@-2 {{method 'invalid9:' declared here}}

- (void)invalid10:(void *__ptrauth(1, 0, 1))a;
// expected-error@-1 {{parameter type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,0,1)'}}
// expected-note@-2 {{method 'invalid10:' declared here}}

@end

@implementation Foo
// expected-warning@-1 2{{method definition for}}

- (void *__ptrauth(1, 1, 1))invalid13 {
// expected-error@-1 {{return type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,1,1)'}}
  return 0;
}

- (void *__ptrauth(1, 0, 1))invalid14 {
// expected-error@-1 {{return type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,0,1)'}}
  return 0;
}

- (void)invalid17:(void *__ptrauth(1, 1, 1))a {
// expected-error@-1 {{parameter type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,1,1)'}}
}

- (void)invalid18:(void *__ptrauth(1, 0, 1))a {
// expected-error@-1 {{parameter type may not be qualified with '__ptrauth'; type is 'void *__ptrauth(1,0,1)'}}
}

@end
