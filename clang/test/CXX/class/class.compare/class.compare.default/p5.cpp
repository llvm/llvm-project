// RUN: %clang_cc1 -std=c++2a -verify %s

// expected-no-diagnostics
namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1}, strong_ordering::equal{0}, strong_ordering::greater{1};
}

// Check that we compare subobjects in the right order.
struct Log {
  char buff[8] = {};
  int n = 0;
  constexpr void add(char c) { buff[n++] = c; }
  constexpr bool operator==(const char *p) const { return __builtin_strcmp(p, buff) == 0; }
};

template<char C> struct B {
  Log *log;
  constexpr bool operator==(const B&) const { log->add(C); return true; }
  constexpr std::strong_ordering operator<=>(const B&) const { log->add(C); return {0}; }
};

struct C : B<'a'>, B<'b'> {
  B<'c'> c;
  B<'d'> d;
  // FIXME: Test arrays once we handle them properly.

  constexpr C(Log *p) : B<'a'>{p}, B<'b'>{p}, c{p}, d{p} {}

  bool operator==(const C&) const = default;
  std::strong_ordering operator<=>(const C&) const = default;
};

constexpr bool check(bool which) {
  Log log;
  C c(&log);
  (void)(which ? c == c : c <=> c);
  return log == "abcd";
}
static_assert(check(false));
static_assert(check(true));
