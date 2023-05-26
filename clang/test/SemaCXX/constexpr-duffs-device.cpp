// RUN: %clang_cc1 -std=c++1y -verify %s
// RUN: %clang_cc1 -std=c++1y -verify -fexperimental-new-constant-interpreter %s

// expected-no-diagnostics
constexpr void copy(const char *from, unsigned long count, char *to) {
        unsigned long n = (count + 7) / 8;
        switch(count % 8) {
        case 0: do {    *to++ = *from++;
        case 7:         *to++ = *from++;
        case 6:         *to++ = *from++;
        case 5:         *to++ = *from++;
        case 4:         *to++ = *from++;
        case 3:         *to++ = *from++;
        case 2:         *to++ = *from++;
        case 1:         *to++ = *from++;
                } while(--n > 0);
        }
}

struct S {
  char stuff[14];
  constexpr S() : stuff{} {
    copy("Hello, world!", 14, stuff);
  }
};

constexpr bool streq(const char *a, const char *b) {
  while (*a && *a == *b) ++a, ++b;
  return *a == *b;
}

static_assert(streq(S().stuff, "Hello, world!"), "should be same");
static_assert(!streq(S().stuff, "Something else"), "should be different");
