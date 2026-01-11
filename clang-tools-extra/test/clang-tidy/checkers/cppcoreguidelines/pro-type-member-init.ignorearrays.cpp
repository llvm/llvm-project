// RUN: %check_clang_tidy %s \
// RUN: cppcoreguidelines-pro-type-member-init %t \
// RUN: -config="{CheckOptions: \
// RUN: {cppcoreguidelines-pro-type-member-init.IgnoreArrays: true}}"

typedef int TypedefArray[4];
using UsingArray = int[4];

struct HasArrayMember {
  HasArrayMember() {}
  // CHECK-MESSAGES: warning: constructor does not initialize these fields: Number
  UsingArray U;
  TypedefArray T;
  int RawArray[4];
  int Number;
};

namespace std {
template <typename T, int N>
struct array {
  T _Elems[N];
  void fill(const T &);
};
}

void test_local_std_array() {
  std::array<int, 4> a;
}

struct OnlyArray {
  int a[4];
};

void test_local_only_array() {
  OnlyArray a;
}

struct Mixed {
  int a[4];
  int b;
};

void test_local_mixed() {
  Mixed m;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: uninitialized record type: 'm'
}

void test_std_array_fill() {
  std::array<char, 10> someArray;
  // CHECK-MESSAGES-NOT: warning: uninitialized record type: 'someArray'
  someArray.fill('n');
}
