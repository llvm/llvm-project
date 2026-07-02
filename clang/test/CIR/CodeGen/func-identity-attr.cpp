// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=TAG

namespace std {
inline namespace __1 {
int *find(int *first, int *last, int value);
template <class Iter, class T> Iter find(Iter first, Iter last, const T &value);
}
struct container {
  int *find(int value);
};
struct traits {
  static int *find(int *first);
};
}

namespace other {
int *find(int *first, int *last, int value);
}

int *std_call(int *first, int *last) { return std::find(first, last, 42); }
// The free std find carries its tag, with inline namespaces looked
// through.
// CHECK: cir.func{{.*}} @_ZNSt3__14find{{.*}} func_info<#cir.func_identity<"std::find">>

struct S {
  void operator()();
};

void other_calls(S &s, std::container &c, int *first, int *last) {
  c.find(1);
  std::traits::find(first);
  other::find(first, last, 42);
  s();
}
char *narrow_call(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPccEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find", narrow_char_params = true>>

char *mixed_call(char *first, char *last, const signed char &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPcaEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>

volatile char *volatile_call(volatile char *first, volatile char *last,
                             const char &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPVccEET_S3_S3_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>

// Members, static members, operators, and functions outside std match no
// entity, so the tagged functions above stay the only tagged functions.
// TAG-COUNT-4: #cir.func_identity
// TAG-NOT: #cir.func_identity
