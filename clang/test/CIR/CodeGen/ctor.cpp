// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Struk {
  int a;
  Struk() {}
};

void baz() {
  Struk s;
}

// CHECK: !rec_Struk = !cir.record<struct "Struk" {!s32i}>

// Note: In the absence of the '-mconstructor-aliases' option, we emit two
//       constructors here. The handling of constructor aliases is currently
//       NYI, but when it is added this test should be updated to add a RUN
//       line that passes '-mconstructor-aliases' to clang_cc1.
// CHECK:   cir.func @_ZN5StrukC2Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK-NEXT:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK-NEXT:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK-NEXT:     cir.return

// CHECK:   cir.func @_ZN5StrukC1Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK-NEXT:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK-NEXT:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK-NEXT:     cir.call @_ZN5StrukC2Ev(%[[THIS]]) : (!cir.ptr<!rec_Struk>) -> ()
// CHECK-NEXT:     cir.return

// CHECK:   cir.func @_Z3bazv()
// CHECK-NEXT:     %[[S_ADDR:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s", init] {alignment = 4 : i64}
// CHECK-NEXT:     cir.call @_ZN5StrukC1Ev(%[[S_ADDR]]) : (!cir.ptr<!rec_Struk>) -> ()
// CHECK-NEXT:     cir.return

struct VariadicStruk {
  int a;
  VariadicStruk(int n, ...) { a = n;}
};

void bar() {
  VariadicStruk s(1, 2, 3);
}

// When a variadic constructor is present, we call the C2 constructor directly.

// CHECK-NOT: cir.func @_ZN13VariadicStrukC2Eiz

// CHECK:      cir.func @_ZN13VariadicStrukC1Eiz(%arg0: !cir.ptr<!rec_VariadicStruk>
// CHECK-SAME:                                   %arg1: !s32i
// CHECK-SAME:                                   ...) {
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   %[[A_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CHECK-NEXT:   cir.store{{.*}} %[[N]], %[[A_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:  cir.func @_Z3barv
// CHECK-NEXT:    %[[S_ADDR:.*]] = cir.alloca !rec_VariadicStruk, !cir.ptr<!rec_VariadicStruk>, ["s", init]
// CHECK-NEXT:    %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:    %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK-NEXT:    %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CHECK-NEXT:    cir.call @_ZN13VariadicStrukC1Eiz(%[[S_ADDR]], %[[ONE]], %[[TWO]], %[[THREE]])
// CHECK-NEXT:    cir.return

struct DelegatingStruk {
  int a;
  DelegatingStruk(int n) { a = n; }
  DelegatingStruk() : DelegatingStruk(0) {}
};

void bam() {
  DelegatingStruk s;
}

// CHECK:       cir.func @_ZN15DelegatingStrukC2Ei(%arg0: !cir.ptr<!rec_DelegatingStruk>
// CHECK-SAME:                                     %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   %[[A_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CHECK-NEXT:   cir.store{{.*}} %[[N]], %[[A_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:       cir.func @_ZN15DelegatingStrukC1Ei(%arg0: !cir.ptr<!rec_DelegatingStruk>
// CHECK-SAME:                                     %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   cir.call @_ZN15DelegatingStrukC2Ei(%[[THIS]], %[[N]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func @_ZN15DelegatingStrukC1Ev(%arg0: !cir.ptr<!rec_DelegatingStruk>
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:   cir.call @_ZN15DelegatingStrukC1Ei(%[[THIS]], %[[ZERO]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func @_Z3bamv
// CHECK-NEXT:    %[[S_ADDR:.*]] = cir.alloca {{.*}} ["s", init]
// CHECK-NEXT:    cir.call @_ZN15DelegatingStrukC1Ev(%[[S_ADDR]])
// CHECK-NEXT:    cir.return
