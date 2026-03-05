// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

struct Foo {
  int a;
};

struct Bar {
  int a;
};

bool eq(int Foo::*x, int Foo::*y) {
  return x == y;
}

// CIR-LABEL: @_Z2eqM3FooiS0_
//      CIR:   %[[#x:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-NEXT:   %[[#y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-NEXT:   %{{.+}} = cir.cmp(eq, %[[#x]], %[[#y]]) : !cir.data_member<!s32i in !rec_Foo>, !cir.bool
//      CIR: }

// LLVM-LABEL: @_Z2eqM3FooiS0_
//      LLVM:   %[[#x:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %[[#y:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.+}} = icmp eq i64 %[[#x]], %[[#y]]
//      LLVM: }

bool ne(int Foo::*x, int Foo::*y) {
  return x != y;
}

// CIR-LABEL: @_Z2neM3FooiS0_
//      CIR:   %[[#x:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-NEXT:   %[[#y:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.data_member<!s32i in !rec_Foo>>, !cir.data_member<!s32i in !rec_Foo>
// CIR-NEXT:   %{{.+}} = cir.cmp(ne, %[[#x]], %[[#y]]) : !cir.data_member<!s32i in !rec_Foo>, !cir.bool
//      CIR: }

// LLVM-LABEL: @_Z2neM3FooiS0_
//      LLVM:   %[[#x:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %[[#y:]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.+}} = icmp ne i64 %[[#x]], %[[#y]]
//      LLVM: }
