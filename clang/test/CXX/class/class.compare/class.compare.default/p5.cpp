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
  char buff[10] = {};
  int n = 0;
  constexpr void add(char c) { buff[n++] = c; }
  constexpr bool operator==(const char *p) const { return __builtin_strcmp(p, buff) == 0; }
};

template<char C> struct B {
  Log *log;
  constexpr bool operator==(const B&) const { log->add(C); return true; }
  constexpr std::strong_ordering operator<=>(const B&) const { log->add(C); return {0}; }
};

template<typename T> constexpr bool check(bool which, const char *str) {
  Log log;
  T c(&log);
  (void)(which ? c == c : c <=> c);
  return log == str;
}

struct C : B<'a'>, B<'b'> {
  B<'r'> r[3];
  B<'c'> c;
  B<'s'> s[2];
  B<'d'> d;

  constexpr C(Log *p) : B<'a'>{p}, B<'b'>{p}, r{p, p, p}, c{p}, s{p, p}, d{p} {}

  bool operator==(const C&) const = default;
  std::strong_ordering operator<=>(const C&) const = default;
};

static_assert(check<C>(false, "abrrrcssd"));
static_assert(check<C>(true, "abrrrcssd"));

struct D {
  B<'x'> x;
  B<'y'> y[2];

  constexpr D(Log *p) : x{p}, y{p, p} {}

  bool operator==(const D&) const = default;
  std::strong_ordering operator<=>(const D&) const = default;
};

static_assert(check<D>(false, "xyy"));
static_assert(check<D>(true, "xyy"));
