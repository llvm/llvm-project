// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1  -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fobjc-arc -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://6386358

#if __has_attribute(objc_requires_super)
#define  NS_REQUIRES_SUPER __attribute((objc_requires_super))
#endif

@protocol NSObject // expected-note {{protocol is declared here}}
- MyDealloc NS_REQUIRES_SUPER; // expected-warning {{'objc_requires_super' attribute cannot be applied to methods in protocols}}
@end

@interface Root
- MyDealloc __attribute((objc_requires_super));
- (void)XXX __attribute((objc_requires_super));
- (void) dealloc __attribute((objc_requires_super)); // expected-warning {{'objc_requires_super' attribute cannot be applied to dealloc}}
- (void) MyDeallocMeth; // Method in root is not annotated.
- (void) AnnotMyDeallocMeth __attribute((objc_requires_super));
- (void) AnnotMyDeallocMethCAT NS_REQUIRES_SUPER; 

+ (void)registerClass:(id)name __attribute((objc_requires_super));
@end

@interface Baz : Root<NSObject>
- MyDealloc;
- (void) MyDeallocMeth __attribute((objc_requires_super)); // 'Baz' author has annotated method
- (void) AnnotMyDeallocMeth; // Annotated in root but not here. Annotation is inherited though
- (void) AnnotMeth __attribute((objc_requires_super)); // 'Baz' author has annotated method
@end

@implementation Baz
-  MyDealloc {
   [super MyDealloc];
        return 0;
}

- (void)XXX {
  [super MyDealloc];
} // expected-warning {{method possibly missing a [super XXX] call}}

- (void) MyDeallocMeth {} // No warning here.
- (void) AnnotMyDeallocMeth{} // expected-warning {{method possibly missing a [super AnnotMyDeallocMeth] call}}
- (void) AnnotMeth{}; // No warning here. Annotation is in its class.

+ (void)registerClass:(id)name {} // expected-warning {{method possibly missing a [super registerClass:] call}}
@end

@interface Bar : Baz
@end

@implementation Bar
- (void) MyDeallocMeth {} // expected-warning {{method possibly missing a [super MyDeallocMeth] call}}
- (void) AnnotMyDeallocMeth{} // expected-warning {{method possibly missing a [super AnnotMyDeallocMeth] call}}
- (void) AnnotMeth{};  // expected-warning {{method possibly missing a [super AnnotMeth] call}}
@end

@interface Bar(CAT) 
- (void) AnnotMyDeallocMethCAT; // Annotated in root but not here. Annotation is inherited though
- (void) AnnotMethCAT __attribute((objc_requires_super));
@end

@implementation Bar(CAT)
- (void) MyDeallocMeth {} // expected-warning {{method possibly missing a [super MyDeallocMeth] call}}
- (void) AnnotMyDeallocMeth{} // expected-warning {{method possibly missing a [super AnnotMyDeallocMeth] call}}
- (void) AnnotMeth{};  // expected-warning {{method possibly missing a [super AnnotMeth] call}}
- (void) AnnotMyDeallocMethCAT{}; // expected-warning {{method possibly missing a [super AnnotMyDeallocMethCAT] call}}
- (void) AnnotMethCAT {};
@end


@interface Valid : Baz
@end

@implementation Valid

- (void)MyDeallocMeth {
  [super MyDeallocMeth]; // no-warning
}


+ (void)registerClass:(id)name {
  [super registerClass:name]; // no-warning
}

@end
