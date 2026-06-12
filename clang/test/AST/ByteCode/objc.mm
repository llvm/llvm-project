// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

@interface A {
  int a;
  static_assert(a, ""); // both-error {{static assertion expression is not an integral constant expression}}
}
@end

@interface NSString
@end
constexpr NSString *t0 = @"abc";
constexpr NSString *t1 = @("abc");


#if __LP64__
typedef unsigned long NSUInteger;
typedef long NSInteger;
#else
typedef unsigned int NSUInteger;
typedef int NSInteger;
#endif


@class NSNumber;


@interface NSObject
+ (NSObject*)nsobject;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithInt:(int)value;
@end

int main(void) {
  NSNumber *bv = @(1391126400 * 1000); // both-warning {{overflow in expression; result is -443'003'904 with type 'int'}}
}
