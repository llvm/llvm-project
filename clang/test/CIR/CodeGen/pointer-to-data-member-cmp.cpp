// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct Foo {
  int a;
};

struct Bar {
  int a;
};

bool eq(int Foo::*x, int Foo::*y) {
  return x == y;
}

// CIR-BEFORE-LABEL: @_Z2eqM3FooiS0_
//      CIR-BEFORE:   %[[#x:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-BEFORE-NEXT:   %[[#y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-BEFORE-NEXT:   %{{.+}} = cir.cmp(eq, %[[#x]], %[[#y]]) : !cir.data_member<!s32i in !rec_Foo>, !cir.bool
//      CIR-BEFORE: }

// CIR-AFTER-LABEL: @_Z2eqM3FooiS0_
// CIR-AFTER:   %[[#x:]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   %[[#y:]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   %{{.*}} = cir.cmp(eq, %[[#x]], %[[#y]]) : !s64i, !cir.bool

// LLVM-LABEL: @_Z2eqM3FooiS0_
//      LLVM:   %[[#x:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %[[#y:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.+}} = icmp eq i64 %[[#x]], %[[#y]]
//      LLVM: }

// OGCG-LABEL: @_Z2eqM3FooiS0_
//      OGCG:   %[[#x:]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:   %[[#y:]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:   %{{.+}} = icmp eq i64 %[[#x]], %[[#y]]
//      OGCG: }

bool ne(int Foo::*x, int Foo::*y) {
  return x != y;
}

// CIR-BEFORE-LABEL: @_Z2neM3FooiS0_
//      CIR-BEFORE:   %[[#x:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-BEFORE-NEXT:   %[[#y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-BEFORE-NEXT:   %{{.+}} = cir.cmp(ne, %[[#x]], %[[#y]]) : !cir.data_member<!s32i in !rec_Foo>, !cir.bool
//      CIR-BEFORE: }

// CIR-AFTER-LABEL: @_Z2neM3FooiS0_
// CIR-AFTER:   %[[#x:]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   %[[#y:]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!s64i>, !s64i
// CIR-AFTER:   %{{.*}} = cir.cmp(ne, %[[#x]], %[[#y]]) : !s64i, !cir.bool

// LLVM-LABEL: @_Z2neM3FooiS0_
//      LLVM:   %[[#x:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %[[#y:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.+}} = icmp ne i64 %[[#x]], %[[#y]]
//      LLVM: }

// OGCG-LABEL: @_Z2neM3FooiS0_
//      OGCG:   %[[#x:]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:   %[[#y:]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:   %{{.+}} = icmp ne i64 %[[#x]], %[[#y]]
//      OGCG: }
