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
// CHECK:   cir.func{{.*}} @_ZN5StrukC2Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK-NEXT:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK-NEXT:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK-NEXT:     cir.return

// CHECK:   cir.func{{.*}} @_ZN5StrukC1Ev(%arg0: !cir.ptr<!rec_Struk>
// CHECK-NEXT:     %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %[[THIS_ADDR]] : !cir.ptr<!rec_Struk>, !cir.ptr<!cir.ptr<!rec_Struk>>
// CHECK-NEXT:     %[[THIS:.*]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_Struk>>, !cir.ptr<!rec_Struk>
// CHECK-NEXT:     cir.call @_ZN5StrukC2Ev(%[[THIS]]) : (!cir.ptr<!rec_Struk>) -> ()
// CHECK-NEXT:     cir.return

// CHECK:   cir.func{{.*}} @_Z3bazv()
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

// CHECK-NOT: cir.func{{.*}} @_ZN13VariadicStrukC2Eiz

// CHECK:      cir.func{{.*}} @_ZN13VariadicStrukC1Eiz(%arg0: !cir.ptr<!rec_VariadicStruk>
// CHECK-SAME:                                   %arg1: !s32i
// CHECK-SAME:                                   ...){{.*}} {
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   %[[A_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CHECK-NEXT:   cir.store{{.*}} %[[N]], %[[A_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:  cir.func{{.*}} @_Z3barv
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

// CHECK:       cir.func{{.*}} @_ZN15DelegatingStrukC2Ei(%arg0: !cir.ptr<!rec_DelegatingStruk>
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

// CHECK:       cir.func{{.*}} @_ZN15DelegatingStrukC1Ei(%arg0: !cir.ptr<!rec_DelegatingStruk>
// CHECK-SAME:                                     %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   cir.call @_ZN15DelegatingStrukC2Ei(%[[THIS]], %[[N]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_ZN15DelegatingStrukC1Ev(%arg0: !cir.ptr<!rec_DelegatingStruk>
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:   cir.call @_ZN15DelegatingStrukC1Ei(%[[THIS]], %[[ZERO]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_Z3bamv
// CHECK-NEXT:    %[[S_ADDR:.*]] = cir.alloca {{.*}} ["s", init]
// CHECK-NEXT:    cir.call @_ZN15DelegatingStrukC1Ev(%[[S_ADDR]])
// CHECK-NEXT:    cir.return

struct MemberInitStruk {
  int a;
  MemberInitStruk() : a(0) {}
};

void init_member() {
  MemberInitStruk s;
}

// CHECK:      cir.func{{.*}} @_ZN15MemberInitStrukC2Ev(%arg0: !cir.ptr<!rec_MemberInitStruk>
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CHECK-NEXT:   %[[A_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CHECK-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:   cir.store align(4) %[[ZERO]], %[[A_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN15MemberInitStrukC1Ev(%arg0: !cir.ptr<!rec_MemberInitStruk>
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CHECK-NEXT:   cir.call @_ZN15MemberInitStrukC2Ev(%[[THIS]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_Z11init_memberv
// CHECK-NEXT:    %[[S_ADDR:.*]] = cir.alloca {{.*}} ["s", init]
// CHECK-NEXT:    cir.call @_ZN15MemberInitStrukC1Ev(%[[S_ADDR]])
// CHECK-NEXT:    cir.return

struct ParamMemberInitStruk {
  int a;
  ParamMemberInitStruk(int n) : a(n) {}
};

void init_param_member() {
  ParamMemberInitStruk s(0);
}

// CHECK:      cir.func{{.*}} @_ZN20ParamMemberInitStrukC2Ei(%arg0: !cir.ptr<!rec_ParamMemberInitStruk>
// CHECK-SAME:                                         %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CHECK-NEXT:   %[[A_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   cir.store{{.*}} %[[N]], %[[A_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN20ParamMemberInitStrukC1Ei(%arg0: !cir.ptr<!rec_ParamMemberInitStruk>
// CHECK-SAME:                                         %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[N_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CHECK-NEXT:   %[[N:.*]] = cir.load{{.*}} %[[N_ADDR]]
// CHECK-NEXT:   cir.call @_ZN20ParamMemberInitStrukC2Ei(%[[THIS]], %[[N]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_Z17init_param_memberv
// CHECK-NEXT:    %[[S_ADDR:.*]] = cir.alloca {{.*}} ["s", init]
// CHECK-NEXT:    %[[ZERO:.*]] = cir.const #cir.int<0>
// CHECK-NEXT:    cir.call @_ZN20ParamMemberInitStrukC1Ei(%[[S_ADDR]], %[[ZERO]])
// CHECK-NEXT:    cir.return

struct UnionInitStruk {
  union {
    int a;
    union {
        float b;
        double c;
    };
  };
  UnionInitStruk() : c(0.0) {}
};

void init_union() {
  UnionInitStruk s;
}

// CHECK:      cir.func{{.*}} @_ZN14UnionInitStrukC2Ev(%arg0: !cir.ptr<!rec_UnionInitStruk>
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CHECK-NEXT:   %[[AU1_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = ""}
// CHECK-NEXT:   %[[AU2_ADDR:.*]] = cir.get_member %[[AU1_ADDR]][1] {name = ""}
// CHECK-NEXT:   %[[C_ADDR:.*]] = cir.get_member %[[AU2_ADDR]][1] {name = "c"}
// CHECK-NEXT:   %[[ZERO:.*]] = cir.const #cir.fp<0.000000e+00>
// CHECK-NEXT:   cir.store{{.*}} %[[ZERO]], %[[C_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN14UnionInitStrukC1Ev(%arg0: !cir.ptr<!rec_UnionInitStruk>
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CHECK-NEXT:   cir.call @_ZN14UnionInitStrukC2Ev
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_Z10init_unionv
// CHECK-NEXT:    %[[S_ADDR:.*]] = cir.alloca {{.*}} ["s", init]
// CHECK-NEXT:    cir.call @_ZN14UnionInitStrukC1Ev(%[[S_ADDR]])
// CHECK-NEXT:    cir.return

struct Base {
  int a;
  Base(int val) : a(val) {}
};

struct Derived : Base {
  Derived(int val) : Base(val) {}
};

void test_derived() {
  Derived d(1);
}

// CHECK: cir.func{{.*}} @_ZN4BaseC2Ei(%arg0: !cir.ptr<!rec_Base> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[VAL_ADDR:.*]] = cir.alloca {{.*}} ["val", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[VAL_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[A_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "a"}
// CHECK-NEXT:   %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]]
// CHECK-NEXT:   cir.store{{.*}} %[[VAL]], %[[A_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN7DerivedC2Ei(%arg0: !cir.ptr<!rec_Derived> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[VAL_ADDR:.*]] = cir.alloca {{.*}} ["val", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[VAL_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CHECK-NEXT:   %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]]
// CHECK-NEXT:   cir.call @_ZN4BaseC2Ei(%[[BASE]], %[[VAL]])
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN7DerivedC1Ei(%arg0: !cir.ptr<!rec_Derived> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[VAL_ADDR:.*]] = cir.alloca {{.*}} ["val", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[VAL_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]]
// CHECK-NEXT:   cir.call @_ZN7DerivedC2Ei(%[[THIS]], %[[VAL]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_Z12test_derivedv
// CHECK-NEXT:    %[[D_ADDR:.*]] = cir.alloca {{.*}} ["d", init]
// CHECK-NEXT:    %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:    cir.call @_ZN7DerivedC1Ei(%[[D_ADDR]], %[[ONE]])
// CHECK-NEXT:    cir.return

struct Base2 {
  int b;
  Base2(int val) : b(val) {}
};

struct Derived2 : Base, Base2 {
  int c;
  Derived2(int val1, int val2, int val3) : Base(val1), Base2(val2), c(val3) {}
};

void test_derived2() {
  Derived2 d(1, 2, 3);
}

// CHECK: cir.func{{.*}} @_ZN5Base2C2Ei(%arg0: !cir.ptr<!rec_Base2> {{.*}}, %arg1: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[VAL_ADDR:.*]] = cir.alloca {{.*}} ["val", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[VAL_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[B_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "b"}
// CHECK-NEXT:   %[[VAL:.*]] = cir.load{{.*}} %[[VAL_ADDR]]
// CHECK-NEXT:   cir.store{{.*}} %[[VAL]], %[[B_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN8Derived2C2Eiii(%arg0: !cir.ptr<!rec_Derived2>
// CHECK-SAME:                                    %arg1: !s32i
// CHECK-SAME:                                    %arg2: !s32i
// CHECK-SAME:                                    %arg3: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[VAL1_ADDR:.*]] = cir.alloca {{.*}} ["val1", init]
// CHECK-NEXT:   %[[VAL2_ADDR:.*]] = cir.alloca {{.*}} ["val2", init]
// CHECK-NEXT:   %[[VAL3_ADDR:.*]] = cir.alloca {{.*}} ["val3", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[VAL1_ADDR]]
// CHECK-NEXT:   cir.store %arg2, %[[VAL2_ADDR]]
// CHECK-NEXT:   cir.store %arg3, %[[VAL3_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived2> nonnull [0] -> !cir.ptr<!rec_Base>
// CHECK-NEXT:   %[[VAL1:.*]] = cir.load{{.*}} %[[VAL1_ADDR]]
// CHECK-NEXT:   cir.call @_ZN4BaseC2Ei(%[[BASE]], %[[VAL1]])
// CHECK-NEXT:   %[[BASE2:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived2> nonnull [4] -> !cir.ptr<!rec_Base2>
// CHECK-NEXT:   %[[VAL2:.*]] = cir.load{{.*}} %[[VAL2_ADDR]]
// CHECK-NEXT:   cir.call @_ZN5Base2C2Ei(%[[BASE2]], %[[VAL2]])
// CHECK-NEXT:   %[[C_ADDR:.*]] = cir.get_member %[[THIS]][2] {name = "c"}
// CHECK-NEXT:   %[[VAL3:.*]] = cir.load{{.*}} %[[VAL3_ADDR]]
// CHECK-NEXT:   cir.store{{.*}} %[[VAL3]], %[[C_ADDR]]
// CHECK-NEXT:   cir.return

// CHECK:      cir.func{{.*}} @_ZN8Derived2C1Eiii(%arg0: !cir.ptr<!rec_Derived2>
// CHECK-SAME:                                    %arg1: !s32i
// CHECK-SAME:                                    %arg2: !s32i
// CHECK-SAME:                                    %arg3: !s32i
// CHECK-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca {{.*}} ["this", init]
// CHECK-NEXT:   %[[VAL1_ADDR:.*]] = cir.alloca {{.*}} ["val1", init]
// CHECK-NEXT:   %[[VAL2_ADDR:.*]] = cir.alloca {{.*}} ["val2", init]
// CHECK-NEXT:   %[[VAL3_ADDR:.*]] = cir.alloca {{.*}} ["val3", init]
// CHECK-NEXT:   cir.store %arg0, %[[THIS_ADDR]]
// CHECK-NEXT:   cir.store %arg1, %[[VAL1_ADDR]]
// CHECK-NEXT:   cir.store %arg2, %[[VAL2_ADDR]]
// CHECK-NEXT:   cir.store %arg3, %[[VAL3_ADDR]]
// CHECK-NEXT:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CHECK-NEXT:   %[[VAL1:.*]] = cir.load{{.*}} %[[VAL1_ADDR]]
// CHECK-NEXT:   %[[VAL2:.*]] = cir.load{{.*}} %[[VAL2_ADDR]]
// CHECK-NEXT:   %[[VAL3:.*]] = cir.load{{.*}} %[[VAL3_ADDR]]
// CHECK-NEXT:   cir.call @_ZN8Derived2C2Eiii(%[[THIS]], %[[VAL1]], %[[VAL2]], %[[VAL3]])
// CHECK-NEXT:   cir.return

// CHECK: cir.func{{.*}} @_Z13test_derived2v
// CHECK-NEXT:    %[[D_ADDR:.*]] = cir.alloca {{.*}} ["d", init]
// CHECK-NEXT:    %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:    %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK-NEXT:    %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CHECK-NEXT:    cir.call @_ZN8Derived2C1Eiii(%[[D_ADDR]], %[[ONE]], %[[TWO]], %[[THREE]])
// CHECK-NEXT:    cir.return
