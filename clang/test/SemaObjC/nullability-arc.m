// RUN: %clang_cc1 -fobjc-arc -fsyntax-only -Woverriding-method-mismatch %s -verify

__attribute__((objc_root_class))
@interface NSFoo
@end

// ARC qualifiers stacked with nullability.
void accepts_arc_qualified(NSFoo * __unsafe_unretained _Nonnull obj) {
  accepts_arc_qualified(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}
