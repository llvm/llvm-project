/* "System header" for testing GNU libc keyword conflict workarounds */

typedef union {
  union w *__uptr;
#if defined(MS) && defined(NOT_SYSTEM)
  // expected-warning@-2 {{keyword '__uptr' will be treated as an identifier here}}
#endif
  int *__iptr;
} WS __attribute__((__transparent_union__));
