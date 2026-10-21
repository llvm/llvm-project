// RUN: %clang_cc1 -fblocks -fsyntax-only -verify=expected,eager %s
// RUN: %clang_cc1 -fblocks -fexperimental-late-parse-attributes -fsyntax-only -verify=expected,late %s

// A counted_by-family attribute on a block pointer is rejected the same way
// with and without late parsing. With late parsing, the attribute used to be
// collected by the `^` declarator chunk and then hit
// `assert(DS.getLateAttributes().empty())` in ParseDeclaratorInternal (a
// release build dropped it silently). It's now handed to the block-pointer
// chunk, so Sema rejects it like any other non-pointer type, and the
// completion pass at the closing brace never parses its argument.

#define __counted_by(f) __attribute__((counted_by(f)))
#define __counted_by_or_null(f) __attribute__((counted_by_or_null(f)))
#define __sized_by(f) __attribute__((sized_by(f)))
#define __sized_by_or_null(f) __attribute__((sized_by_or_null(f)))

struct count_before {
  int n;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  void (^__counted_by(n) a)(void);
  // expected-error@+1{{'counted_by_or_null' only applies to pointers}}
  void (^__counted_by_or_null(n) b)(void);
  // expected-error@+1{{'sized_by' only applies to pointers}}
  void (^__sized_by(n) c)(void);
  // expected-error@+1{{'sized_by_or_null' only applies to pointers}}
  void (^__sized_by_or_null(n) d)(void);
};

// With late parsing, the rejected attribute's argument is never parsed, so a
// count declared after the field doesn't add an "undeclared identifier" error.
struct count_after {
  // late-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // eager-error@+1{{use of undeclared identifier 'n'}}
  void (^__counted_by(n) a)(void);
  int n;
};
