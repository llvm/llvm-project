
// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

void pass_to_bidi_indexable(struct flexible *__unsafe_indexable flex) {
  // expected-error@+1{{initializing 'struct flexible *__bidi_indexable' with an expression of incompatible type 'struct flexible *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  struct flexible *b = flex;
}

void pass_to_single(struct flexible *__unsafe_indexable flex) {
  // expected-error@+1{{initializing 'struct flexible *__single' with an expression of incompatible type 'struct flexible *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  struct flexible *__single s = flex;
}

void pass_to_single_forge(struct flexible *__unsafe_indexable flex) {
  struct flexible *__single s = __unsafe_forge_single(struct flexible *, flex);
}

void pass_to_single2(struct flexible *__unsafe_indexable flex) {
  struct flexible *__single s;
  // expected-error@+1{{assigning to 'struct flexible *__single' from incompatible type 'struct flexible *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  s = flex;
}

void pass_to_single_assign(struct flexible *__unsafe_indexable flex) {
  // expected-error@+1{{initializing 'struct flexible *__single' with an expression of incompatible type 'struct flexible *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  struct flexible *__single s = flex;
  s->count = flex->count;
}

struct flex_unsafe {
    int count;
    int elems[];
};

void promote_unsafe_to_bidi_indexable(struct flex_unsafe *__unsafe_indexable flex) {
  // expected-error@+1{{initializing 'struct flex_unsafe *__bidi_indexable' with an expression of incompatible type 'struct flex_unsafe *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  struct flex_unsafe *b = flex;
}

void promote_unsafe_to_single(struct flex_unsafe *__unsafe_indexable flex) {
  // expected-error@+1{{initializing 'struct flex_unsafe *__bidi_indexable' with an expression of incompatible type 'struct flex_unsafe *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  struct flex_unsafe *s = flex;
}

void promote_unsafe_to_unsafe(struct flex_unsafe *__unsafe_indexable flex) {
  struct flex_unsafe *__unsafe_indexable s = flex;
}
