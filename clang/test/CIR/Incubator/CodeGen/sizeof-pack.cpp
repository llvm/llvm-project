// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test basic sizeof... on type parameter pack
template<typename ...Types>
int get_num_types(Types...) {
  return sizeof...(Types);
}

// CHECK-LABEL: cir.func{{.*}} @{{.*}}get_num_typesIJifdEEiDpT_
// CHECK: %{{.*}} = cir.const #cir.int<3> : !u64i
// CHECK: %{{.*}} = cir.cast integral %{{.*}} : !u64i -> !s32i

template int get_num_types(int, float, double);

// Test sizeof... with empty pack
template<typename ...Types>
int get_num_empty(Types...) {
  return sizeof...(Types);
}

// CHECK-LABEL: cir.func{{.*}} @{{.*}}get_num_emptyIJEEiDpT_
// CHECK: %{{.*}} = cir.const #cir.int<0> : !u64i

template int get_num_empty();

// Test sizeof... on non-type parameter pack
template<int... Vals>
int count_values() {
  return sizeof...(Vals);
}

// CHECK-LABEL: cir.func{{.*}} @{{.*}}count_valuesIJLi1ELi2ELi3ELi4ELi5EEEiv
// CHECK: %{{.*}} = cir.const #cir.int<5> : !u64i

template int count_values<1, 2, 3, 4, 5>();
