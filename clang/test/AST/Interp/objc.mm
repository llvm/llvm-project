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
