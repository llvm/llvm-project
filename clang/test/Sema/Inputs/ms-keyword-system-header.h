/* "System header" for testing GNU libc keyword conflict workarounds */

typedef union {
  union w *__uptr;
  int *__iptr;
} WS __attribute__((__transparent_union__));
