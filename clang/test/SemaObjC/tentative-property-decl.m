// RUN: %clang_cc1 -fsyntax-only -Weverything -verify %s
// rdar://11656982
/** Normally, a property cannot be both 'readonly' and having a "write" attribute
    (copy/retain/etc.). But, property declaration in primary class and protcols
    are tentative as they may be overridden into a 'readwrite' property in class 
    extensions. Postpone diagnosing such warnings until the class implementation 
    is seen.
*/

@interface Super {
}
@end

@class NSString;

@interface MyClass : Super
@property(nonatomic, copy, readonly) NSString *prop;
@property(nonatomic, copy, readonly) id warnProp; // expected-warning {{property attributes 'readonly' and 'copy' are mutually exclusive}}
@end

@interface MyClass ()
@property(nonatomic, copy, readwrite) NSString *prop;
@end

@implementation MyClass
@synthesize prop;
@synthesize warnProp;
@end


@protocol P
@property(nonatomic, copy, readonly) NSString *prop;
@property(nonatomic, copy, readonly) id warnProp; // expected-warning {{property attributes 'readonly' and 'copy' are mutually exclusive}}
@end

@interface YourClass : Super <P>
@end

@interface YourClass ()
@property(nonatomic, copy, readwrite) NSString *prop;
@end

@implementation YourClass 
@synthesize prop;
@synthesize warnProp;
@end

