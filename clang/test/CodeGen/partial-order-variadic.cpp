// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclang-abi-compat=14 -DCLANG_ABI_COMPAT=14 %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-14
// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

#if defined(CLANG_ABI_COMPAT) && CLANG_ABI_COMPAT <= 14

// CHECK-14: define dso_local void @_ZN24temp_func_order_example31hEi(i32 noundef %i)
// CHECK-14-NEXT: entry:
// CHECK-14-NEXT:   %i.addr = alloca i32, align 4
// CHECK-14-NEXT:   %r = alloca ptr, align 8
// CHECK-14-NEXT:   store i32 %i, ptr %i.addr, align 4
// CHECK-14-NEXT:   %call = call noundef nonnull align 4 dereferenceable(4) ptr @_ZN24temp_func_order_example31gIiJEEERiPT_DpT0_(ptr noundef %i.addr)
// CHECK-14-NEXT:   store ptr %call, ptr %r, align 8
// CHECK-14-NEXT:   ret void

namespace temp_func_order_example3 {
  template <typename T, typename... U> int &g(T *, U...);
  template <typename T> void g(T);
  void h(int i) {
    int &r = g(&i);
  }
}

#else

// CHECK: %"struct.temp_deduct_type_example1::A" = type { i8 }

// CHECK: $_ZN25temp_deduct_type_example31fIiJEEEvPT_DpT0_ = comdat any

// CHECK: define dso_local void @_ZN25temp_deduct_type_example11fEv()
// CHECK-NEXT: entry:
// CHECK-NEXT:   %a = alloca %"struct.temp_deduct_type_example1::A", align 1
// CHECK-NEXT:   ret void

// CHECK: define weak_odr void @_ZN25temp_deduct_type_example31fIiJEEEvPT_DpT0_(ptr noundef %0)
// CHECK-NEXT: entry:
// CHECK-NEXT:   %.addr = alloca ptr, align 8
// CHECK-NEXT:   store ptr %0, ptr %.addr, align 8
// CHECK-NEXT:   ret void

namespace temp_deduct_type_example1 {
  template<class T, class... U> struct A;
  template<class T1, class T2, class... U> struct A<T1,T2*,U...> {};
  template<class T1, class T2> struct A<T1,T2>;
  template struct A<int, int*>;
  void f() { A<int, int*> a; }
}

namespace temp_deduct_type_example3 {
  template<class T, class... U> void f(T*, U...){}
  template<class T> void f(T){}
  template void f(int*);
}

#endif
