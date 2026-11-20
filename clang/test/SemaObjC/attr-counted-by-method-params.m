// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

int global_count;

// An Objective-C method parameter's count is checked like a function
// parameter's.
@interface I
// A count inside a callback or block parameter names that parameter's own
// parameters.
- (void)callback:(void (*)(int len, int *__counted_by(len) p))cb;
- (void)callback_or_null:(void (*)(int len, int *__counted_by_or_null(len) p))cb;
- (void)callback_sized:(void (*)(int len, void *__sized_by(len) p))cb;
- (void)callback_sized_or_null:(void (*)(int len, void *__sized_by_or_null(len) p))cb;
- (void)block:(void (^)(int len, int *__counted_by(len) p))blk;

// expected-error@+1{{count expression in function declaration may only reference function parameters}}
- (void)global:(int *__counted_by(global_count))buf;
// FIXME: Accept this. The method's parameters are declared only after all of
// them are parsed, so 'n' is not in scope yet.
// expected-error@+1{{use of undeclared identifier 'n'}}
- (void)count:(int)n buf:(int *__counted_by(n))buf;
@end
