// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-objc-root-class -Wdocumentation -verify %s
// expected-no-diagnostics

@interface NSPredicate
///     The full predicate to be used for drawing objects from the store.
///     It is an AND of the parent's `prefixPredicate` (e.g., the selection for
///     volume number) and the `filterPredicate` (selection by matching the name).
///     @return `nil` if there is no search string, and no prefix.

@property(readonly) NSPredicate *andPredicate;
///     The predicate that matches the string to be searched for. This
///     @return `nil` if there is no search string.
@property(readonly) NSPredicate *filterPredicate;
@end
