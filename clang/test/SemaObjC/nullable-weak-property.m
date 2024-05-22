// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -Wnullable-to-nonnull-conversion %s -verify

@interface NSObject @end

@class NSFoo;
void foo (NSFoo * _Nonnull);

@interface NSBar : NSObject
@property(weak) NSFoo *property1;
@end

#pragma clang assume_nonnull begin
@interface NSBar ()
@property(weak) NSFoo *property2;
@end

#pragma clang assume_nonnull end

@implementation NSBar 
- (void) Meth {
   foo (self.property1); // no warning because nothing is inferred
   foo (self.property2); // expected-warning {{implicit conversion from nullable pointer 'NSFoo * _Nullable' to non-nullable pointer type 'NSFoo * _Nonnull'}}
}
@end
