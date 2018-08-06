// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s  \
// RUN:     -faligned-allocation -fsized-deallocation -emit-llvm -o - \
// RUN:    | FileCheck %s

typedef __SIZE_TYPE__ size_t;

// Declare an 'operator new' template to tickle a bug in __builtin_operator_new.
template<typename T> void *operator new(size_t, int (*)(T));

// Ensure that this declaration doesn't cause operator new to lose its
// 'noalias' attribute.
void *operator new(size_t);

namespace std {
  struct nothrow_t {};
  enum class align_val_t : size_t { __zero = 0,
                                  __max = (size_t)-1 };
}
std::nothrow_t nothrow;

// Declare the reserved placement operators.
void *operator new(size_t, void*) throw();
void operator delete(void*, void*) throw();
void *operator new[](size_t, void*) throw();
void operator delete[](void*, void*) throw();

// Declare the replaceable global allocation operators.
void *operator new(size_t, const std::nothrow_t &) throw();
void *operator new[](size_t, const std::nothrow_t &) throw();
void operator delete(void *, const std::nothrow_t &) throw();
void operator delete[](void *, const std::nothrow_t &) throw();

// Declare some other placement operators.
void *operator new(size_t, void*, bool) throw();
void *operator new[](size_t, void*, bool) throw();


// CHECK-LABEL: define void @test_basic(
extern "C" void test_basic() {
  // CHECK: call i8* @_Znwm(i64 4) [[ATTR_BUILTIN_NEW:#[^ ]*]]
  // CHECK: call void @_ZdlPv({{.*}}) [[ATTR_BUILTIN_DELETE:#[^ ]*]]
  // CHECK: ret void
  __builtin_operator_delete(__builtin_operator_new(4));
}
// CHECK: declare noalias i8* @_Znwm(i64) [[ATTR_NOBUILTIN:#[^ ]*]]
// CHECK: declare void @_ZdlPv(i8*) [[ATTR_NOBUILTIN_NOUNWIND:#[^ ]*]]

// CHECK-LABEL: define void @test_aligned_alloc(
extern "C" void test_aligned_alloc() {
  // CHECK: call i8* @_ZnwmSt11align_val_t(i64 4, i64 4) [[ATTR_BUILTIN_NEW:#[^ ]*]]
  // CHECK: call void @_ZdlPvSt11align_val_t({{.*}}, i64 4) [[ATTR_BUILTIN_DELETE:#[^ ]*]]
  __builtin_operator_delete(__builtin_operator_new(4, std::align_val_t(4)), std::align_val_t(4));
}
// CHECK: declare noalias i8* @_ZnwmSt11align_val_t(i64, i64) [[ATTR_NOBUILTIN:#[^ ]*]]
// CHECK: declare void @_ZdlPvSt11align_val_t(i8*, i64) [[ATTR_NOBUILTIN_NOUNWIND:#[^ ]*]]


// CHECK-LABEL: define void @test_sized_delete(
extern "C" void test_sized_delete() {
  // CHECK: call i8* @_Znwm(i64 4) [[ATTR_BUILTIN_NEW:#[^ ]*]]
  // CHECK: call void @_ZdlPvm({{.*}}, i64 4) [[ATTR_BUILTIN_DELETE:#[^ ]*]]
  __builtin_operator_delete(__builtin_operator_new(4), 4);
}
// CHECK: declare void @_ZdlPvm(i8*, i64) [[ATTR_NOBUILTIN_UNWIND:#[^ ]*]]


// CHECK-DAG: attributes [[ATTR_NOBUILTIN]] = {{[{].*}} nobuiltin {{.*[}]}}
// CHECK-DAG: attributes [[ATTR_NOBUILTIN_NOUNWIND]] = {{[{].*}} nobuiltin nounwind {{.*[}]}}

// CHECK-DAG: attributes [[ATTR_BUILTIN_NEW]] = {{[{].*}} builtin {{.*[}]}}
// CHECK-DAG: attributes [[ATTR_BUILTIN_DELETE]] = {{[{].*}} builtin {{.*[}]}}
