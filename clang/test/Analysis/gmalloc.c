// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-store=region -verify %s

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
void g_free(gpointer mem);
gpointer g_memdup(gconstpointer mem, guint byte_size);

static const gsize n_bytes = 1024;

void f1() {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);

  g_free(g1);
  g_free(g2);
  g_free(g2); // expected-warning{{Attempt to free released memory}}
}

void f2() {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2);

  g_free(g1);
  g_free(g2);
  g_free(g3);
  g3 = g_memdup(g3, n_bytes); // expected-warning{{Use of memory after it is freed}}
}

void f3() {
  gpointer g1 = g_malloc(n_bytes);
  gpointer g2 = g_malloc0(n_bytes);
  g1 = g_realloc(g1, n_bytes * 2);
  gpointer g3 = g_try_malloc(n_bytes);
  gpointer g4 = g_try_malloc0(n_bytes);
  g3 = g_try_realloc(g3, n_bytes * 2); // expected-warning{{Potential leak of memory pointed to by 'g4'}}

  g_free(g1);
  g_free(g2);
  g_free(g3);
}
