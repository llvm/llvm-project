// RUN: %clang_cc1 -fsyntax-only -verify %s

@class MyObject;


@interface TopClassWithClassProperty0
@property(nullable, readonly, strong, class) MyObject *foo;
@end

@interface SubClassWithClassProperty0 : TopClassWithClassProperty0
@property(nonnull, readonly, copy, class) MyObject *foo; // expected-warning {{'copy' attribute on property 'foo' does not match the property inherited from 'TopClassWithClassProperty0'}}
@end



@interface TopClassWithInstanceProperty1
@property(nullable, readonly, strong) MyObject *foo;
@end

@interface ClassWithClassProperty1 : TopClassWithInstanceProperty1
@property(nonnull, readonly, copy, class) MyObject *foo; // no-warning
@end

@interface SubClassWithInstanceProperty1 : ClassWithClassProperty1
@property(nullable, readonly, copy) MyObject *foo; // expected-warning {{'copy' attribute on property 'foo' does not match the property inherited from 'TopClassWithInstanceProperty1'}}
@end


@interface TopClassWithClassProperty2
@property(nullable, readonly, strong, class) MyObject *foo;
@end

@interface ClassWithInstanceProperty2 : TopClassWithClassProperty2
@property(nonnull, readonly, copy) MyObject *foo; // no-warning
@end

@interface SubClassWithClassProperty2 : ClassWithInstanceProperty2
@property(nonnull, readonly, copy, class) MyObject *foo; // expected-warning {{'copy' attribute on property 'foo' does not match the property inherited from 'TopClassWithClassProperty2'}}
@end
