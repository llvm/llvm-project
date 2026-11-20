// RUN: %clang_cc1 -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

int global_count;

// An Objective-C method parameter's count is checked like a function
// parameter's.
@interface I
// expected-error@+1{{count expression in function declaration may only reference function parameters}}
- (void)global:(int *__counted_by(global_count))buf;
// The method's earlier parameters are not in scope yet.
// expected-error@+1{{use of undeclared identifier 'n'}}
- (void)count:(int)n buf:(int *__counted_by(n))buf;
@end
