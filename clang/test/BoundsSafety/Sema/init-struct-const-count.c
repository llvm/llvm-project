// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fbounds-safety-bringup-missing-checks=all -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fno-bounds-safety-bringup-missing-checks=all -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fbounds-safety-bringup-missing-checks=all -verify %s
#include <ptrcheck.h>

// expected-no-diagnostics

// =============================================================================
// __counted_by
// =============================================================================

struct cb {
  const int count;
  int* __counted_by(count) ptr;
};

void consume_cb(struct cb);

void init_list_cb(int count_param, int*__counted_by(count_param) ptr) {
  struct cb c = {.count = count_param, .ptr = ptr };
  consume_cb(c);
}

void init_list_cb_bidi(int count_param, int* __bidi_indexable ptr) {
  struct cb c = {.count = count_param, .ptr = ptr };
  consume_cb(c);
}

void compound_literal_init_cb(int count_param, int*__counted_by(count_param) ptr) {
  struct cb c = (struct cb){.count = count_param, .ptr = ptr };
  consume_cb(c);
}

void compound_literal_init_cb_bidi(int count_param, int*__bidi_indexable ptr) {
  struct cb c = (struct cb){.count = count_param, .ptr = ptr };
  consume_cb(c);
}

// =============================================================================
// __counted_by_or_null
// =============================================================================

struct cbon {
  const int count;
  int* __counted_by_or_null(count) ptr;
};

void consume_cbon(struct cbon);

void init_list_cbon(int count_param, int*__counted_by_or_null(count_param) ptr) {
  struct cbon c = {.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

void init_list_cbon_bidi(int count_param, int*__bidi_indexable ptr) {
  struct cbon c = {.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

void compound_literal_init_cbon(int count_param, int*__counted_by_or_null(count_param) ptr) {
  struct cbon c = (struct cbon){.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

void compound_literal_init_cbon_bidi(int count_param, int*__bidi_indexable ptr) {
  struct cbon c = (struct cbon){.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

// =============================================================================
// __sized_by
// =============================================================================

struct sb {
  const int count;
  char* __sized_by(count) ptr;
};

void consume_sb(struct sb);

void init_list_sb(int count_param, char*__sized_by(count_param) ptr) {
  struct sb c = {.count = count_param, .ptr = ptr };
  consume_sb(c);
}

void init_list_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sb c = {.count = count_param, .ptr = ptr };
  consume_sb(c);
}

void compound_literal_init_sb(int count_param, char*__sized_by(count_param) ptr) {
  struct sb c = (struct sb){.count = count_param, .ptr = ptr };
  consume_sb(c);
}

void compound_literal_init_sb_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sb c = (struct sb){.count = count_param, .ptr = ptr };
  consume_sb(c);
}

// =============================================================================
// __sized_by_or_null
// =============================================================================

struct sbon {
  const int count;
  char* __sized_by_or_null(count) ptr;
};

void consume_sbon(struct sbon);

void init_list_sbon(int count_param, char*__sized_by_or_null(count_param) ptr) {
  struct sbon c = {.count = count_param, .ptr = ptr };
  consume_sbon(c);
}

void init_list_sbon_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sbon c = {.count = count_param, .ptr = ptr };
  consume_sbon(c);
}

void compound_literal_init_sbon(int count_param, char*__sized_by_or_null(count_param) ptr) {
  struct sbon c = (struct sbon){.count = count_param, .ptr = ptr };
  consume_sbon(c);
}

void compound_literal_init_sbon_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sbon c = (struct sbon){.count = count_param, .ptr = ptr };
  consume_sbon(c);
}
