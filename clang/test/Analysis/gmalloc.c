// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -verify %s

#include "Inputs/system-header-simulator.h"

typedef void* gpointer;
typedef const void* gconstpointer;
typedef unsigned long gsize;
typedef unsigned int guint;

gpointer g_malloc(gsize n_bytes);
gpointer g_malloc0(gsize n_bytes);
gpointer g_realloc(gpointer mem, gsize n_bytes);
gpointer g_try_malloc(gsize n_bytes);
gpointer g_try_malloc0(gsize n_bytes);
gpointer g_try_realloc(gpointer mem, gsize n_bytes);
gpointer g_malloc_n(gsize n_blocks, gsize n_block_bytes);
gpointer g_malloc0_n(gsize n_blocks, gsize n_block_bytes);
gpointer g_realloc_n(gpointer mem, gsize n_blocks, gsize n_block_bytes);
gpointer g_try_malloc_n(gsize n_blocks, gsize n_block_bytes);
gpointer g_try_malloc0_n(gsize n_blocks, gsize n_block_bytes);
gpointer g_try_realloc_n(gpointer mem, gsize n_blocks, gsize n_block_bytes);
void g_free(gpointer mem);
gpointer g_memdup(gconstpointer mem, guint byte_size);
gpointer g_strconcat(gconstpointer string1, ...);

static const gsize n_bytes = 1024;

void f1(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char));
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char));
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char));

  g_free(g1);
  g_free(g2);
  g_free(g2); // expected-warning{{Attempt to free released memory}}
}

void f2(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char));
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char));
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char));

  g_free(g1);
  g_free(g2);
  g_free(g3);
  g3 = g_memdup(g3, n_bytes); // expected-warning{{Use of memory after it is freed}}
}

void f3(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2); // expected-warning{{Potential leak of memory pointed to by 'g4'}}
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g6'}}
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g5'}}
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g8'}}

  g_free(g1); // expected-warning{{Potential leak of memory pointed to by 'g7'}}
  g_free(g2);
  g_free(g3);
}

void f4(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g6'}}
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g5'}}
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g8'}}

  g_free(g1); // expected-warning{{Potential leak of memory pointed to by 'g7'}}
  g_free(g2);
  g_free(g3);
  g_free(g4);
}

void f5(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g6'}}
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char));
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g8'}}

  g_free(g1); // expected-warning{{Potential leak of memory pointed to by 'g7'}}
  g_free(g2);
  g_free(g3);
  g_free(g4);
  g_free(g5);
}

void f6(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char));
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char));
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g8'}}

  g_free(g1); // expected-warning{{Potential leak of memory pointed to by 'g7'}}
  g_free(g2);
  g_free(g3);
  g_free(g4);
  g_free(g5);
  g_free(g6);
}

void f7(void) {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);
  gpointer g5 = g_malloc_n(n_bytes, sizeof(char));
  gpointer g6 = g_malloc0_n(n_bytes, sizeof(char));
  g5 = g_realloc_n(g5, n_bytes * 2, sizeof(char));
  gpointer g7 = g_try_malloc_n(n_bytes, sizeof(char));
  gpointer g8 = g_try_malloc0_n(n_bytes, sizeof(char));
  g7 = g_try_realloc_n(g7, n_bytes * 2, sizeof(char)); // expected-warning{{Potential leak of memory pointed to by 'g8'}}

  g_free(g1);
  g_free(g2);
  g_free(g3);
  g_free(g4);
  g_free(g5);
  g_free(g6);
  g_free(g7);
}

void f8(void) {
  typedef struct {
    gpointer str;
  } test_struct;

  test_struct *s1 = (test_struct *)g_malloc0(sizeof(test_struct));
  test_struct *s2 = (test_struct *)g_memdup(s1, sizeof(test_struct));
  gpointer str = g_strconcat("text", s1->str, s2->str, NULL); // no-warning
  g_free(str);
  g_free(s2);
  g_free(s1);
}
