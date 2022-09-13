// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

constexpr int m = 3;
constexpr const int *foo[][5] = {
  {nullptr, &m, nullptr, nullptr, nullptr},
  {nullptr, nullptr, &m, nullptr, nullptr},
  {nullptr, nullptr, nullptr, &m, nullptr},
};

static_assert(foo[0][0] == nullptr, "");
static_assert(foo[0][1] == &m, "");
static_assert(foo[0][2] == nullptr, "");
static_assert(foo[0][3] == nullptr, "");
static_assert(foo[0][4] == nullptr, "");
static_assert(foo[1][0] == nullptr, "");
static_assert(foo[1][1] == nullptr, "");
static_assert(foo[1][2] == &m, "");
static_assert(foo[1][3] == nullptr, "");
static_assert(foo[1][4] == nullptr, "");
static_assert(foo[2][0] == nullptr, "");
static_assert(foo[2][1] == nullptr, "");
static_assert(foo[2][2] == nullptr, "");
static_assert(foo[2][3] == &m, "");
static_assert(foo[2][4] == nullptr, "");


/// A init list for a primitive value.
constexpr int f{5};
static_assert(f == 5, "");


constexpr int getElement(int i) {
  int values[] = {1, 4, 9, 16, 25, 36};
  return values[i];
}
static_assert(getElement(1) == 4, "");
static_assert(getElement(5) == 36, "");


template<typename T>
constexpr T getElementOf(T* array, int i) {
  return array[i];
}
static_assert(getElementOf(foo[0], 1) == &m, "");


template <typename T, int N>
constexpr T& getElementOfArray(T (&array)[N], int I) {
  return array[I];
}
static_assert(getElementOfArray(foo[2], 3) == &m, "");


constexpr int data[] = {5, 4, 3, 2, 1};
static_assert(data[0] == 4, ""); // expected-error{{failed}} \
                                 // expected-note{{5 == 4}} \
                                 // ref-error{{failed}} \
                                 // ref-note{{5 == 4}}


constexpr int dynamic[] = {
  f, 3, 2 + 5, data[3], *getElementOf(foo[2], 3)
};
static_assert(dynamic[0] == f, "");
static_assert(dynamic[3] == 2, "");


constexpr int dependent[4] = {
  0, 1, dependent[0], dependent[1]
};
static_assert(dependent[2] == dependent[0], "");
static_assert(dependent[3] == dependent[1], "");

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Winitializer-overrides"
constexpr int DI[] = {
  [0] = 10,
  [1] = 20,
  30,
  40,
  [1] = 50
};
static_assert(DI[0] == 10, "");
static_assert(DI[1] == 50, "");
static_assert(DI[2] == 30, "");
static_assert(DI[3] == 40, "");

/// FIXME: The example below tests ImplicitValueInitExprs, but we can't
///   currently evaluate other parts of it.
#if 0
struct fred {
  char s [6];
  int n;
};

struct fred y [] = { [0] = { .s[0] = 'q' } };
#endif
#pragma clang diagnostic pop
