// RUN: %clang_cc1 -fsyntax-only -Weverything -verify %s
// expected-no-diagnostics
// rdar://11656982
/** A property may not be both 'readonly' and having a memory management attribute
    (copy/retain/etc.). But, property declaration in primary class and protcols
    are tentative as they may be overridden into a 'readwrite' property in class 
    extensions. So, do not issue any warning on 'readonly' and memory management
    attributes in a property.
*/

@interface Super {
}
@end

@class NSString;

@interface MyClass : Super
@property(nonatomic, copy, readonly) NSString *prop;
@property(nonatomic, copy, readonly) id warnProp;
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
@property(nonatomic, copy, readonly) id warnProp;
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

