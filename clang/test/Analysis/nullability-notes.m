// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core \
// RUN:   -analyzer-checker=nullability.NullPassedToNonnull \
// RUN:   -analyzer-checker=nullability.NullReturnedFromNonnull \
// RUN:   -analyzer-checker=nullability.NullablePassedToNonnull \
// RUN:   -analyzer-checker=nullability.NullableReturnedFromNonnull \
// RUN:   -analyzer-checker=nullability.NullableDereferenced \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core \
// RUN:   -analyzer-checker=nullability.NullPassedToNonnull \
// RUN:   -analyzer-checker=nullability.NullReturnedFromNonnull \
// RUN:   -analyzer-checker=nullability.NullablePassedToNonnull \
// RUN:   -analyzer-checker=nullability.NullableReturnedFromNonnull \
// RUN:   -analyzer-checker=nullability.NullableDereferenced \
// RUN:   -analyzer-output=plist -o %t.plist %s
// RUN: %normalize_plist <%t.plist \
// RUN:   | diff -ub %S/Inputs/expected-plists/nullability-notes.m.plist -

void clang_analyzer_warnOnDeadSymbol(id);

#include "Inputs/system-header-simulator-for-nullability.h"

void takesNonnull(NSObject *_Nonnull y);

@interface ClassWithProperties: NSObject
@property(copy, nullable) NSObject *x; // plist check ensures no control flow piece from here to 'self.x'.
-(void) method;
@end;
@implementation ClassWithProperties
-(void) method {
  clang_analyzer_warnOnDeadSymbol(self);
  // no-crash
  NSObject *x = self.x; // expected-note{{Nullability 'nullable' is inferred}}
                        // expected-warning@-1{{SYMBOL DEAD}}
                        // expected-note@-2   {{SYMBOL DEAD}}
  takesNonnull(x); // expected-warning{{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
                   // expected-note@-1{{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
}
@end

