// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface I1 {
@private
  int x;
  struct {
    unsigned int x : 3;
    unsigned int y : 3;
  } flags;
  int y;
}
@end
