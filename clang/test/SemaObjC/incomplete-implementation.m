// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I
- Meth; // expected-note 2 {{method 'Meth' declared here}}
- unavailableMeth __attribute__((availability(macosx,unavailable)));
- unavailableMeth2 __attribute__((unavailable));
@end

@implementation  I  // expected-warning {{method definition for 'Meth' not found}}
@end

@implementation I(CAT)
- Meth {return 0;} // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end

@interface MyClass
-(void)mymeth __attribute__((availability(macos, introduced=100))); // expected-note{{here}}
@end
@implementation MyClass // expected-warning{{'mymeth' not found}}
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

@implementation Q

__attribute__((visibility("default")))
@interface QN // expected-error {{Objective-C declarations may only appear in global scope}}
{
}
@end

@end

typedef char BOOL;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@interface NSObject <NSObject>
@end

@protocol NSApplicationDelegate <NSObject>
- (void)ImpleThisMethod; // expected-note {{method 'ImpleThisMethod' declared here}}
@end

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate (MRRCategory)

- (BOOL)isEqual:(id)object
{
    return __objc_no;
}

- (void)ImpleThisMethod {} // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end
