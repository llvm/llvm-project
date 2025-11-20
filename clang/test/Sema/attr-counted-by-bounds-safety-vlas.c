// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety -fexperimental-late-parse-attributes -verify %s
//
// This is a portion of the `attr-counted-by-vla.c` test but is checked
// under the semantics of `-fexperimental-bounds-safety` which has different
// behavior.

#define __counted_by(f)  __attribute__((counted_by(f)))

struct has_unannotated_VLA {
  int count;
  char buffer[];
};

struct has_annotated_VLA {
  int count;
  char buffer[] __counted_by(count);
};

struct buffer_of_structs_with_unnannotated_vla {
  int count;
  // expected-error@+1{{'counted_by' cannot be applied to an array with element of unknown size because 'struct has_unannotated_VLA' is a struct type with a flexible array member}}
  struct has_unannotated_VLA Arr[] __counted_by(count);
};


struct buffer_of_structs_with_annotated_vla {
  int count;
  // expected-error@+1{{'counted_by' cannot be applied to an array with element of unknown size because 'struct has_annotated_VLA' is a struct type with a flexible array member}}
  struct has_annotated_VLA Arr[] __counted_by(count);
};

struct buffer_of_const_structs_with_annotated_vla {
  int count;
  // Make sure the `const` qualifier is printed when printing the element type.
  // expected-error@+1{{'counted_by' cannot be applied to an array with element of unknown size because 'const struct has_annotated_VLA' is a struct type with a flexible array member}}
  const struct has_annotated_VLA Arr[] __counted_by(count);
};
