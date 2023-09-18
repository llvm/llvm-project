// Objective-C recovery
// RUN: cp %s %t
// RUN: %clang_cc1 -Wmissing-method-return-type -fixit -x objective-c -Wno-objc-root-class %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wmissing-method-return-type -Werror -x objective-c -Wno-objc-root-class %t

// Objective-C++ recovery
// RUN: cp %s %t
// RUN: %clang_cc1 -Wmissing-method-return-type -fixit -x objective-c++ -Wno-objc-root-class %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Wmissing-method-return-type -Werror -x objective-c++ -Wno-objc-root-class %t

@interface I
-  initWithFoo:(id)foo; // expected-warning {{method has no return type specified; defaults to 'id' [-Wmissing-method-return-type]}}
- Meth;
-Meth1;
@end

@implementation I
- initWithFoo:(id)foo { return 0; } // expected-warning {{method has no return type specified; defaults to 'id' [-Wmissing-method-return-type]}}

-Meth { return 0;}
- Meth1 { return 0;}
@end

