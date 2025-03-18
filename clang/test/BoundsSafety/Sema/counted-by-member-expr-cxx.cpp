// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s

#include <ptrcheck.h>
#include <stddef.h>

typedef struct T {
    size_t count;
} T;

class C {
  T t;
  T *tp;
  size_t count;
  int * __counted_by(count) p;       // implicit `this->count` is ok
  int * __counted_by(this->count) q; // explicit `this->count` is ok
  int * __counted_by(t.count) x;     // expected-error{{invalid argument expression to bounds attribute}} expected-note{{nested struct member in count parameter only supported for flexible array members}}
  int * __counted_by(tp->count) y;   // expected-error{{invalid argument expression to bounds attribute}} expected-note{{nested struct member in count parameter only supported for flexible array members}}
};

// test for simple flexible array members:
typedef struct flexible {
    size_t count;
    int elems[__counted_by(count)];
} flex_t;

class FAM {
  size_t count;
  int fam[__counted_by(count)];
public:
  FAM() {};
};

class FAM_DOT {
  T t;
  int fam[__counted_by(t.count)];     // dot-expressions in counted-by is ok for FAMs
};

class FAM_ARROW {
  T *tp;
  int fam[__counted_by(tp->count)]; // expected-error{{arrow notation not allowed for struct member in count parameter}}
};

class FAM_THIS_ARROW_ARROW {
  T *tp;
  int fam[__counted_by(this->tp->count)]; // expected-error{{arrow notation not allowed for struct member in count parameter}}
};

class FAM_THIS_ARROW_DOT {
  T t;
  int fam[__counted_by(this->t.count)]; // dot-expressions in counted-by is ok for FAMs
};

class FAM_ARITHMETIC {
  int count;
  int offset;
  int fam[__counted_by(count - offset)]; //  ok
};

class FAM_THIS_PTR_ARITHMETIC {
  int count;
  int fam[__counted_by((this + 1)->count)]; // expected-error{{arrow notation not allowed for struct member in count parameter}}
};

class FAM_THIS_PTR_DEREFERENCE {
  int count;
  int fam[__counted_by((*this).count)];     // expected-error{{invalid argument expression to bounds attribute}}
};
