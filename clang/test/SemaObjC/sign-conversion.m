// RUN: %clang_cc1 -fsyntax-only -verify -Wsign-conversion %s
// rdar://13855394

typedef unsigned int NSUInteger;

@interface NSObject
- new;
- (NSUInteger)hash;
@end

@interface X : NSObject
@property NSUInteger uint;
@end

@interface NSArray : NSObject 

- (NSUInteger)count;
- (id)objectAtIndex:(NSUInteger)index;
- (id)objectAtIndexedSubscript:(NSUInteger)index;

@end

void foo(void) {
    X *x = [X new];
    signed int sint = -1;
    [x setUint:sint];  // expected-warning {{implicit conversion changes signedness: 'int' to 'NSUInteger'}}
    x.uint = sint; // expected-warning {{implicit conversion changes signedness: 'int' to 'NSUInteger'}}
}

// rdar://13855682
void Test1(void) {
signed int si = -1;
NSArray *array;

(void)((NSObject*)array[si]).hash; // expected-warning {{implicit conversion changes signedness: 'int' to 'NSUInteger'}}

(void)[((NSObject*)array[si]) hash]; // expected-warning {{implicit conversion changes signedness: 'int' to 'NSUInteger'}}
(void)array[si]; // expected-warning {{implicit conversion changes signedness: 'int' to 'NSUInteger'}}
}
