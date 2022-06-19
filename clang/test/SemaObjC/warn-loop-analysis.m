// RUN: %clang_cc1 -fsyntax-only -Wloop-analysis -verify %s
// expected-no-diagnostics

@interface MyArray
- (id)objectAtIndexedSubscript:(unsigned int)idx;
@end

// Do not warn on objc classes has objectAtIndexedSubscript method.
MyArray *test;
void foo(void)
{
  unsigned int i;
  for (i = 42; i > 0;) // No warnings here
    (void)test[--i];
}
