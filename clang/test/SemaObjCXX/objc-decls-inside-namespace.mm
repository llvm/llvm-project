// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

namespace C {

@protocol P; //expected-error{{Objective-C declarations may only appear in global scope}}

@class Bar; //expected-error{{Objective-C declarations may only appear in global scope}}

@compatibility_alias Foo Bar; //expected-error{{Objective-C declarations may only appear in global scope}}

@interface A //expected-error{{Objective-C declarations may only appear in global scope}}
@end

@implementation A //expected-error{{Objective-C declarations may only appear in global scope}}
@end

@protocol P //expected-error{{Objective-C declarations may only appear in global scope}}
@end

@interface A(C) //expected-error{{Objective-C declarations may only appear in global scope}}
@end

@implementation A(C) //expected-error{{Objective-C declarations may only appear in global scope}}
@end

@interface B @end //expected-error{{Objective-C declarations may only appear in global scope}}
@implementation B //expected-error{{Objective-C declarations may only appear in global scope}}
+ (void) foo {}
@end

}

