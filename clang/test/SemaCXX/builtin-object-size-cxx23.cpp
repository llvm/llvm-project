// RUN: %clang_cc1 -fsyntax-only -std=c++23 %s

constexpr int stringLength(const char *p) {
  return __builtin_dynamic_object_size(p, 0);
}

static_assert(stringLength("hello") == 6);

constexpr int allocation(unsigned n) {
  const char * ptr = new char[n];
  int res = stringLength(ptr);
  delete[] ptr;
  return res;
}

static_assert(allocation(1) == 1);
static_assert(allocation(14) == 14);



