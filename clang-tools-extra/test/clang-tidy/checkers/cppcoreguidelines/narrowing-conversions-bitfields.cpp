// RUN: %check_clang_tidy %s cppcoreguidelines-narrowing-conversions %t \
// RUN:   -std=c++17 -- -target x86_64-unknown-linux

#define CHAR_BITS 8
static_assert(sizeof(unsigned int) == 32 / CHAR_BITS);

template <typename T, typename U>
struct is_same {
  static constexpr bool value = false;
};
template <typename T>
struct is_same<T, T> {
  static constexpr bool value = true;
};

template <typename T, typename U>
static constexpr bool is_same_v = is_same<T, U>::value;

struct NoBitfield {
  unsigned int id;
};
struct SmallBitfield {
  unsigned int id : 4;
};

struct BigBitfield {
  unsigned int id : 31;
};
struct CompleteBitfield {
  unsigned int id : 32;
};

int example_warning(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  return x;
}

void test_binary_and(SmallBitfield x) {
  static_assert(is_same_v<decltype(x.id & 1), int>);
  static_assert(is_same_v<decltype(x.id & 1u), unsigned>);

  x.id & 1;
  x.id & 1u;

  1 & x.id;
  1u & x.id;
}

void test_binary_or(SmallBitfield x) {
  static_assert(is_same_v<decltype(x.id | 1), int>);
  static_assert(is_same_v<decltype(x.id | 1u), unsigned>);

  x.id | 1;
  x.id | 1u;

  1 | x.id;
  1u | x.id;
}

template <typename T>
void take(T);

void test_parameter_passing(NoBitfield x) {
  take<char>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: narrowing conversion from 'unsigned int' to signed type 'char' is implementation-defined
  take<short>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'unsigned int' to signed type 'short' is implementation-defined
  take<unsigned>(x.id);
  take<int>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined
  take<long>(x.id);
  take<long long>(x.id);
}

void test_parameter_passing(SmallBitfield x) {
  take<char>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: narrowing conversion from 'unsigned int' to signed type 'char' is implementation-defined
  take<short>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'unsigned int' to signed type 'short' is implementation-defined
  take<unsigned>(x.id);
  take<int>(x.id); // no-warning
  take<long>(x.id);
  take<long long>(x.id);
}

void test_parameter_passing(BigBitfield x) {
  take<char>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: narrowing conversion from 'unsigned int' to signed type 'char' is implementation-defined
  take<short>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'unsigned int' to signed type 'short' is implementation-defined
  take<unsigned>(x.id);
  take<int>(x.id); // no-warning
  take<long>(x.id);
  take<long long>(x.id);
}

void test_parameter_passing(CompleteBitfield x) {
  take<char>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: narrowing conversion from 'unsigned int' to signed type 'char' is implementation-defined
  take<short>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'unsigned int' to signed type 'short' is implementation-defined
  take<unsigned>(x.id);
  take<int>(x.id);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined
  take<long>(x.id);
  take<long long>(x.id);
}

void test(NoBitfield x) {
  static_assert(is_same_v<decltype(x.id << 1), unsigned>);
  static_assert(is_same_v<decltype(x.id << 1u), unsigned>);
  static_assert(is_same_v<decltype(x.id + 1), unsigned>);
  static_assert(is_same_v<decltype(x.id + 1u), unsigned>);

  x.id << 1;
  x.id << 1u;
  x.id >> 1;
  x.id >> 1u;
  x.id + 1;
  x.id + 1u;

  1 << x.id;
  1u << x.id;
  1 >> x.id;
  1u >> x.id;
  1 + x.id;
  1u + x.id;
}

void test(SmallBitfield x) {
  static_assert(is_same_v<decltype(x.id << 1), int>);
  static_assert(is_same_v<decltype(x.id << 1u), int>);

  x.id << 1;
  x.id << 1u;
  x.id >> 1;
  x.id >> 1u;

  x.id + 1;
  x.id + 1u;

  1 << x.id;
  1u << x.id;
  1 >> x.id;
  1u >> x.id;

  1 + x.id;
  1u + x.id;
}

void test(BigBitfield x) {
  static_assert(is_same_v<decltype(x.id << 1), int>);
  static_assert(is_same_v<decltype(x.id << 1u), int>);

  x.id << 1;
  x.id << 1u;
  x.id >> 1;
  x.id >> 1u;

  x.id + 1;
  x.id + 1u;

  1 << x.id;
  1u << x.id;
  1 >> x.id;
  1u >> x.id;

  1 + x.id;
  1u + x.id;
}

void test(CompleteBitfield x) {
  static_assert(is_same_v<decltype(x.id << 1), unsigned>);
  static_assert(is_same_v<decltype(x.id << 1u), unsigned>);

  x.id << 1;
  x.id << 1u;
  x.id >> 1;
  x.id >> 1u;

  x.id + 1;
  x.id + 1u;

  1 << x.id;
  1u << x.id;
  1 >> x.id;
  1u >> x.id;

  1 + x.id;
  1u + x.id;
}

void test_parens(SmallBitfield x) {
  static_assert(is_same_v<decltype(x.id << (2)), int>);
  static_assert(is_same_v<decltype(((x.id)) << (2)), int>);
  x.id << (2);
  ((x.id)) << (2);

  static_assert(is_same_v<decltype((2) << x.id), int>);
  static_assert(is_same_v<decltype((2) << ((x.id))), int>);
  (2) << x.id;
  (2) << ((x.id));
}
