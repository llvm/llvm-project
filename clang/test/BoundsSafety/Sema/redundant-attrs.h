#define bidiPtr2_header int *__bidi_indexable
#define intPtr2_header int *
#define BIDI_INDEXABLE_header(T, X)    do { T __bidi_indexable X; } while (0)
#define BIDI_INDEXABLE2_header(X)    do { int * __bidi_indexable X; } while (0)
#define nullTermPtr2_header int *__null_terminated
#define nullTermPtr3_header int * __attribute__((__bidi_indexable__))
