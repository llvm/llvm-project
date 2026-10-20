// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// Redundant parens between a rejected counted_by-family attribute and the
// declarator chunk that rejects it. With late parsing, the rejected type used
// to be dropped along with the ParenType around it, so the type no longer
// matched the declarator's chunks and building its TypeLoc asserted
// (`Chunk.Kind == DeclaratorChunk::Pointer` in VisitPointerTypeLoc).

#define __counted_by(f) __attribute__((counted_by(f)))

extern int g;

struct nested_in_parens {
  int n;
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int *__counted_by(n) (*p);
  // expected-error@+1{{'counted_by' attribute on nested pointer type is not allowed}}
  int *__counted_by(n) ((*pp));
};

struct return_type_in_parens {
  int n;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a function return type}}
  int *__counted_by(n) ((*fp)(void));
};

// The parens group the function declarator, so they sit between the counted
// pointer and the function chunk.
// expected-error@+1{{'counted_by' attribute cannot be applied to a function return type}}
int *__counted_by(g) (fn(void));
