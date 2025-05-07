
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

typedef struct flexible {
    int count;
    int elems[__counted_by(count)]; // \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}}
} flex_t;

// just to make sure this is OK
flex_t zeroes = {};
flex_t zeroes2 = { 0 };
flex_t zeroes3 = { .count = 0 };
struct flexible flex = { 3, {1, 2, 3} }; 
struct flexible flex2 = { .count = 3, .elems = { 1, 2, 3} };
struct flexible flex3 = {3, { [2] = 3 } };
struct flexible flex4 = { .count = 3, .elems = { [2] = 3} };
struct flexible *returning_flexible_ptr(void);
flex_t *returning_flex_ptr(void);
void accepting_flexible_ptr(struct flexible *p);
void accepting_flex_ptr(flex_t *p);

// these shouldn't work
struct flexible returning_flexible(void); // expected-error{{-fbounds-safety forbids passing 'struct flexible' by copy}}
flex_t returning_flex(void); // expected-error{{-fbounds-safety forbids passing 'flex_t' (aka 'struct flexible') by copy}}
void accepting_flexible(struct flexible p); // expected-error{{-fbounds-safety forbids passing 'struct flexible' by copy}}
void accepting_flex(flex_t p); // expected-error{{-fbounds-safety forbids passing 'flex_t' (aka 'struct flexible') by copy}}

flex_t negative_count_die = { .count = -1, {1} }; // expected-error{{flexible array member is initialized with 1 element, but count value is initialized to -1}}
flex_t negative_count_die2 = { .count = -1 }; // expected-error{{flexible array member is initialized with 0 elements, but count value is initialized to -1}}
flex_t negative_count = { -1, {1} }; // expected-error{{flexible array member is initialized with 1 element, but count value is initialized to -1}}
flex_t zero_count = { 0, {1, 2, 3 } }; // expected-error{{flexible array member is initialized with 3 elements, but count value is initialized to 0}}
flex_t count_too_small = { 2, {1, 2, 3} }; // expected-error{{flexible array member is initialized with 3 elements, but count value is initialized to 2}}
flex_t count_too_large = {4, {1, 2, 3} }; // expected-error{{flexible array member is initialized with 3 elements, but count value is initialized to 4}}
flex_t count_too_large_2 = {3}; // expected-error{{flexible array member is initialized with 0 elements, but count value is initialized to 3}}
flex_t count_too_large_die = { .count = 3 }; // expected-error{{flexible array member is initialized with 0 elements, but count value is initialized to 3}}
flex_t count_too_large_die2 = { .count = 3, .elems = { 0 } }; // expected-error{{flexible array member is initialized with 1 element, but count value is initialized to 3}}
flex_t count_too_large_die3 = { .count = 3, .elems = { [1] = 0 } }; // expected-error{{flexible array member is initialized with 2 elements, but count value is initialized to 3}}

void foo(void) {
    flex_t flex_local = flex; // expected-error{{-fbounds-safety forbids passing 'struct flexible' by copy}}
    flex_local = flex; // expected-error{{-fbounds-safety forbids passing 'struct flexible' by copy}}
}

flex_t *bar(flex_t *__bidi_indexable flex) {
    ++flex; // expected-error{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
    flex++; // expected-error{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
    --flex; // expected-error{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
    flex--; // expected-error{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
    (void) (flex + 1); // expected-error{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
    (void) (flex - 1); // expected-error{{-fbounds-safety forbids arithmetic on pointers to types with a flexible array member}}
}

void baz(flex_t *__counted_by(1) flex); // expected-error{{cannot apply '__counted_by' attribute to 'flex_t *' (aka 'struct flexible *') because 'flex_t' (aka 'struct flexible') has unknown size; did you mean to use '__sized_by' instead?}}
void qux(flex_t *__sized_by(siz) flex, unsigned siz);

void quux(flex_t *flex, char *__bidi_indexable buf) {
  flex = (flex_t *)buf; // OK. run-time check inserted with flex->count.
}

void quuz(flex_t *flex, char *__bidi_indexable buf) {
  flex = (flex_t *)buf; // OK.
  flex->count = 10;
}

void corge(flex_t *flex, char *__bidi_indexable buf) {
  flex->count = 10; // expected-error{{assignment to 'flex->count' requires an immediately preceding assignment to 'flex' with a wide pointer}}
  flex = (flex_t *)buf;
}

void grault(flex_t *flex) {
  flex->count = 10; // expected-error{{assignment to 'flex->count' requires an immediately preceding assignment to 'flex' with a wide pointer}}
}

void garply(char *__bidi_indexable buf) {
  flex_t *__single flex = (flex_t *)buf;
  flex->count = 10;
}

void waldo(char *__bidi_indexable buf) {
  flex_t *flex;
  flex->count = 10; // OK.
}

void fred(flex_t *flex, char *__bidi_indexable buf) {
  flex = (flex_t *)buf;
  int a = 5;
  flex->count = a; // expected-error{{assignment to 'flex->count' requires an immediately preceding assignment to 'flex' with a wide pointer}}
}

flex_t g_flex = {4, {1, 2, 3, 4}};
void global_flex_count_assign(unsigned new_count) {
  g_flex.count = new_count; // run-time check
}
void global_flex_count_increment() {
  g_flex.count++; // expected-error{{incrementing 'g_flex.count' always traps}}
}

void global_flex_count_decrement() {
  g_flex.count--; // ok. run-time check
}

void global_flex_count_compound_assign(unsigned diff) {
  g_flex.count += diff; // run-time check
}

typedef struct flexible_unannotated {
    int count;
    int elems[]; // \
    // expected-note{{initialized flexible array member 'elems' is here}} \
    // expected-note{{initialized flexible array member 'elems' is here}}
} flex_bad_t;

flex_bad_t bad1 = { 0 }; // expected-error{{flexible array member is initialized without a count}}
flex_bad_t bad2 = {1, {1} }; // expected-error{{flexible array member is initialized without a count}}
