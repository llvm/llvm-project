// RUN: %clang_cc1 %s -triple arm64-apple-macosx     -fsized-deallocation    -faligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx  -fno-sized-deallocation    -faligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx  -fno-sized-deallocation -fno-aligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx     -fsized-deallocation -fno-aligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o - | FileCheck %s

// RUN: %clang_cc1 %s -triple arm64-apple-macosx -faligned-allocation -fexperimental-cxx-type-aware-allocators -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -fexperimental-new-constant-interpreter -o - | FileCheck %s

using size_t = __SIZE_TYPE__;

namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : size_t {};
}


template <typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
template <typename T> void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t);
struct S {
  int i = 0;
  constexpr S() __attribute__((noinline)) {}
};

 constexpr int doSomething() {
  S* s = new S;
  int result = s->i;
  delete s;
  return result;
}

static constexpr int force_doSomething = doSomething();
template <int N> struct Tag {};

void test1(Tag<force_doSomething>){
// CHECK-LABEL: define void @_Z5test13TagILi0EE
}

void test2(Tag<doSomething() + 1>){
// CHECK-LABEL: define void @_Z5test23TagILi1EE
}

int main() {
  // CHECK-LABEL: define noundef i32 @main()
  return doSomething();
  // CHECK: call{{.*}}i32 @_Z11doSomethingv()
}

// CHECK-LABEL: define linkonce_odr noundef i32 @_Z11doSomethingv()
// CHECK: [[ALLOC:%.*]] = call noundef ptr @_ZnwI1SEPvSt13type_identityIT_Em
// CHECK: invoke noundef ptr @_ZN1SC1Ev(ptr{{.*}} [[ALLOC]])
// CHECK-NEXT: to label %[[CONT:.*]] unwind label %[[LPAD:[a-z0-9]+]]
// CHECK: [[CONT]]:
// CHECK: call void @_ZdlI1SEvSt13type_identityIT_EPv
// CHECK: ret
// CHECK: [[LPAD]]:
// call void @_ZdlI1SEvSt13type_identityIT_EPvmSt11align_val_t({{.*}} [[ALLOC]])
