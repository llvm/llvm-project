// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -fms-compatibility -std=c++17 -emit-llvm -o - %s | FileCheck %s

// Dynamic initializers of inline static data members must run in declaration
// order. In the MS ABI, Clang used to emit in-class initialized static data
// members of integral type as soon as the class definition was complete, which
// placed their initializers ahead of preceding members of class type.
// See https://github.com/llvm/llvm-project/issues/110975.

struct A {
  A(int);
  int get() const;
};

struct S {
  static const inline A a = A(1);
  static const inline int b = a.get();
  static inline A c = A(b);
  static inline int d = c.get();
};

struct Outer {
  static inline A e = A(2);
  struct Inner {
    static inline int f = e.get();
  };
};

template <typename> struct Spec;
template <> struct Spec<int> {
  static inline A g = A(3);
  static inline int h = g.get();
};

// Unreferenced members of a dllexport class must still be emitted.
struct __declspec(dllexport) Exported {
  static inline A i = A(4);
  static inline int j = i.get();
};

// CHECK-DAG: @"?i@Exported@@2UA@@A" = weak_odr dso_local dllexport global %struct.A
// CHECK-DAG: @"?j@Exported@@2HA" = weak_odr dso_local dllexport global i32

// CHECK:      @llvm.global_ctors = appending global [10 x { i32, ptr, ptr }]
// CHECK-SAME: { i32 65535, ptr @"??__E?a@S@@2UA@@B@@YAXXZ", ptr @"?a@S@@2UA@@B" }
// CHECK-SAME: { i32 65535, ptr @"??__E?b@S@@2HB@@YAXXZ", ptr @"?b@S@@2HB" }
// CHECK-SAME: { i32 65535, ptr @"??__E?c@S@@2UA@@A@@YAXXZ", ptr @"?c@S@@2UA@@A" }
// CHECK-SAME: { i32 65535, ptr @"??__E?d@S@@2HA@@YAXXZ", ptr @"?d@S@@2HA" }
// CHECK-SAME: { i32 65535, ptr @"??__E?e@Outer@@2UA@@A@@YAXXZ", ptr @"?e@Outer@@2UA@@A" }
// CHECK-SAME: { i32 65535, ptr @"??__E?f@Inner@Outer@@2HA@@YAXXZ", ptr @"?f@Inner@Outer@@2HA" }
// CHECK-SAME: { i32 65535, ptr @"??__E?g@?$Spec@H@@2UA@@A@@YAXXZ", ptr @"?g@?$Spec@H@@2UA@@A" }
// CHECK-SAME: { i32 65535, ptr @"??__E?h@?$Spec@H@@2HA@@YAXXZ", ptr @"?h@?$Spec@H@@2HA" }
// CHECK-SAME: { i32 65535, ptr @"??__E?i@Exported@@2UA@@A@@YAXXZ", ptr @"?i@Exported@@2UA@@A" }
// CHECK-SAME: { i32 65535, ptr @"??__E?j@Exported@@2HA@@YAXXZ", ptr @"?j@Exported@@2HA" }]
