// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializednessComplete=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false

typedef __typeof(sizeof(int)) size_t;
typedef __WCHAR_TYPE__ wchar_t;
typedef __CHAR16_TYPE__ char16_t;
typedef long time_t;
typedef struct {
  int x;
  int y;
} mbstate_t;
struct tm {
  int x;
  int y;
};
extern size_t mbrlen(const char *restrict, size_t, mbstate_t *restrict);
extern size_t wcsnrtombs(char *restrict dst, const wchar_t **restrict src,
       size_t nwc, size_t len, mbstate_t *restrict ps);
extern size_t mbrtoc16(char16_t *restrict pc16, const char *restrict s,
       size_t n, mbstate_t *restrict ps);
extern time_t mktime(struct tm *timeptr);

void uninit_mbrlen(const char *mbs) {
  mbstate_t state;
  mbrlen(mbs, 1, &state); // expected-warning{{3rd function call argument points to an uninitialized value}}
}

void init_mbrlen(const char *mbs) {
  mbstate_t state;
  state.x = 0;
  mbrlen(mbs, 1, &state);
}

void uninit_wcsnrtombs(const wchar_t *src) {
  char dst[10];
  mbstate_t state;
  wcsnrtombs(dst, &src, 1, 2, &state); // expected-warning{{5th function call argument points to an uninitialized value}}
}

void uninit_mbrtoc16(const char *s) {
  char16_t pc16[10];
  mbstate_t state;
  mbrtoc16(pc16, s, 1, &state); // expected-warning{{4th function call argument points to an uninitialized value}}
}

void uninit_mktime() {
  struct tm time;
  mktime(&time); // expected-warning{{1st function call argument points to an uninitialized value}}
}
