// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s

@class NSString;

@protocol MyProtocol
@property (nonatomic, strong, readonly) NSString *myString;
@end

@interface MyClass <MyProtocol>
// Don't warn about this setter:
@property (nonatomic, strong, setter=setMYString:) NSString *myString;


@property (nonatomic, strong, readonly) NSString *overridenInClass; // expected-note {{property declared here}}
@end

@interface MySubClass: MyClass
@property (nonatomic, strong, setter=setMYOverride:) NSString *overridenInClass;
// expected-warning@-1 {{'setter' attribute on property 'overridenInClass' does not match the property inherited from 'MyClass'}}
@end
