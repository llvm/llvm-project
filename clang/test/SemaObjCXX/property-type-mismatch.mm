// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol P1;

@interface NSObject
@end

@interface A : NSObject
@property (assign) NSObject<P1> *prop;
@end

@protocol P2 <P1>
@end

@interface B : A
@property (assign) NSObject<P2> *prop;
@end

@interface C<T> : NSObject 
@end

@interface D
@property (nonatomic,readonly,nonnull) C<D *> *property;
@end

@interface D ()
@property (nonatomic, setter=_setProperty:) C *property; // okay
@end
