// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I
- Meth; // expected-note{{method definition for 'Meth' not found}} \
        // expected-note{{method 'Meth' declared here}}
@end

@implementation  I  // expected-warning{{incomplete implementation}}
@end

@implementation I(CAT)
- Meth {return 0;} // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end

#pragma GCC diagnostic ignored "-Wincomplete-implementation"
@interface I2
- Meth; // expected-note{{method 'Meth' declared here}}
@end

@implementation  I2
@end

@implementation I2(CAT)
- Meth {return 0;} // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end

@interface Q
@end

// rdar://10336158
@implementation Q

__attribute__((visibility("default")))
@interface QN // expected-error {{Objective-C declarations may only appear in global scope}}
{
}
@end

@end

