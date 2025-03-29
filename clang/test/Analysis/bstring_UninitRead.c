// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core,alpha.unix.cstring

//===----------------------------------------------------------------------===//
// mempcpy() using character array. This is the easiest case, as memcpy
// intepretrs the dst and src buffers as character arrays (regardless of their
// actual type).
//===----------------------------------------------------------------------===//

typedef typeof(sizeof(int)) size_t;

void clang_analyzer_eval(int);

void *memcpy(void *restrict s1, const void *restrict s2, size_t n);

void memcpy_array_fully_uninit(char *dst) {
  char buf[10];
  memcpy(dst, buf, 10); // expected-warning{{The first element of the 2nd argument is undefined}}
                        // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

void memcpy_array_partially_uninit(char *dst) {
  char buf[10];
  buf[0] = 'i';
  memcpy(dst, buf, 10); // expected-warning{{The last accessed element (at index 9) in the 2nd argument is undefined}}
                        // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

void memcpy_array_only_init_portion(char *dst) {
  char buf[10];
  buf[0] = 'i';
  memcpy(dst, buf, 1);
  (void)buf;
}

void memcpy_array_partially_init_error(char *dst) {
  char buf[10];
  buf[0] = 'i';
  memcpy(dst, buf, 2); // expected-warning{{The last accessed element (at index 1) in the 2nd argument is undefined}}
                      // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

// The interesting case here is that the portion we're copying is initialized,
// but not the whole matrix. We need to be careful to extract buf[1], and not
// buf when trying to peel region layers off from the source argument.
void memcpy_array_from_matrix(char *dst) {
  char buf[2][2];
  buf[1][0] = 'i';
  buf[1][1] = 'j';
  // FIXME: This is a FP -- we mistakenly retrieve the first element of buf,
  // instead of the first element of buf[1]. getLValueElement simply peels off
  // another ElementRegion layer, when in this case it really shouldn't.
  memcpy(dst, buf[1], 2); // expected-warning{{The first element of the 2nd argument is undefined}}
                          // expected-note@-1{{Other elements might also be undefined}}
  (void)buf;
}

//===----------------------------------------------------------------------===//
// mempcpy() using non-character arrays.
//===----------------------------------------------------------------------===//

void *mempcpy(void *restrict s1, const void *restrict s2, size_t n);

void memcpy_int_array_fully_init() {
  int src[] = {1, 2, 3, 4};
  int dst[5] = {0};
  int *p;

  p = mempcpy(dst, src, 4 * sizeof(int));
  clang_analyzer_eval(p == &dst[4]);
}

void memcpy_int_array_fully_init2(int *dest) {
  int t[] = {1, 2, 3};
  memcpy(dest, t, sizeof(t));
}

//===----------------------------------------------------------------------===//
// mempcpy() using nonarrays.
//===----------------------------------------------------------------------===//

struct st {
  int i;
  int j;
};

void mempcpy_struct_partially_uninit() {
  struct st s1 = {0};
  struct st s2;
  struct st *p1;
  struct st *p2;

  p1 = (&s2) + 1;

  // FIXME: Maybe ask UninitializedObjectChecker whether s1 is fully
  // initialized?
  p2 = mempcpy(&s2, &s1, sizeof(struct st));

  clang_analyzer_eval(p1 == p2);
}

void mempcpy_struct_fully_uninit() {
  struct st s1;
  struct st s2;

  // FIXME: Maybe ask UninitializedObjectChecker whether s1 is fully
  // initialized?
  mempcpy(&s2, &s1, sizeof(struct st));
}

// Creduced crash. In this case, an symbolicregion is wrapped in an
// elementregion for the src argument.
void *ga_copy_strings_from_0;
void *memmove();
void alloc();
void ga_copy_strings() {
  int i = 0;
  for (;; ++i)
    memmove(alloc, ((char **)ga_copy_strings_from_0)[i], 1);
}

// Creduced crash. In this case, retrieving the Loc for the first element failed.
char mov_mdhd_language_map[][4] = {};
int ff_mov_lang_to_iso639_code;
char *ff_mov_lang_to_iso639_to;
void ff_mov_lang_to_iso639() {
  memcpy(ff_mov_lang_to_iso639_to,
         mov_mdhd_language_map[ff_mov_lang_to_iso639_code], 4);
}
