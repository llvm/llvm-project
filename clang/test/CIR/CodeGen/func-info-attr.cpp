// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=NEG

namespace std {
int *find(int *first, int *last, int value);
inline namespace __1 {
void sort(int *first, int *last);
}
struct container {
  int *begin();
};
struct traits {
  static int *locate(int *first);
};
}

int *std_call(int *first, int *last) { return std::find(first, last, 42); }
// Functions declared in the std namespace carry their source identity.
// CHECK-DAG: cir.func{{.*}} @_ZSt4find{{.*}} func_info<#cir.func_info<name = "find", in_std_namespace = true>>

void inline_ns_call(int *first, int *last) { std::sort(first, last); }
// Inline namespaces, like the one in libc++, are looked through.
// CHECK-DAG: cir.func{{.*}} @_ZNSt3__14sort{{.*}} func_info<#cir.func_info<name = "sort", in_std_namespace = true>>

int *member_call(std::container &c) { return c.begin(); }
// Member functions of std records are marked as instance methods.
// CHECK-DAG: cir.func{{.*}} @_ZNSt9container5begin{{.*}} func_info<#cir.func_info<name = "begin", in_std_namespace = true, instance_method = true>>

struct S {
  void method();
  void operator()();
};

void plain_calls(S &s) {
  s.method();
  s();
}
int *static_member_call(int *first) { return std::traits::locate(first); }
// Functions outside the std namespace carry nothing, and neither do
// functions without a plain name, such as operators, nor static member
// functions, even inside std.
// NEG-NOT: name = "method"
// NEG-NOT: name = "operator
// NEG-NOT: name = "locate"
