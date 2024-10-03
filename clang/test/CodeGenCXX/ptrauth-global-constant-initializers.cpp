// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls -fno-rtti -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fno-rtti -fptrauth-vtable-pointer-type-discrimination \
// RUN:   -fptrauth-vtable-pointer-address-discrimination -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ELF

// CHECK: %struct.Base1 = type { ptr }
// CHECK: %struct.Base2 = type { ptr }
// CHECK: %struct.Derived1 = type { %struct.Base1, %struct.Base2 }
// CHECK: %struct.Derived2 = type { %struct.Base2, %struct.Base1 }
// CHECK: %struct.Derived3 = type { %struct.Base1, %struct.Base2 }

// CHECK: @_ZTV5Base1 = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC:38871]], ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV5Base1, i32 0, i32 0, i32 2))] },{{.*}} align 8
// CHECK: @g_b1 = global %struct.Base1 { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV5Base1, i32 0, i32 0, i32 2), i32 2, i64 [[BASE1_VTABLE_DISC:6511]], ptr @g_b1) },{{.*}} align 8
// CHECK: @_ZTV5Base2 = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC:27651]], ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV5Base2, i32 0, i32 0, i32 2))] },{{.*}} align 8
// CHECK: @g_b2 = global %struct.Base2 { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV5Base2, i32 0, i32 0, i32 2), i32 2, i64 [[BASE2_VTABLE_DISC:63631]], ptr @g_b2) },{{.*}} align 8
// CHECK: @_ZTV8Derived1 = linkonce_odr unnamed_addr constant { [5 x ptr], [3 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived1, i32 0, i32 0, i32 2)), ptr ptrauth (ptr @_ZN8Derived11cEv, i32 0, i64 [[DERIVED1_C_DISC:54092]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived1, i32 0, i32 0, i32 3)), ptr ptrauth (ptr @_ZN8Derived11dEv, i32 0, i64 [[DERIVED1_D_DISC:37391]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived1, i32 0, i32 0, i32 4))], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived1, i32 0, i32 1, i32 2))] },{{.*}} align 8
// CHECK: @g_d1 = global { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived1, i32 0, i32 0, i32 2), i32 2, i64 [[BASE1_VTABLE_DISC]], ptr @g_d1), ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived1, i32 0, i32 1, i32 2), i32 2, i64 [[BASE2_VTABLE_DISC]], ptr getelementptr inbounds ({ ptr, ptr }, ptr @g_d1, i32 0, i32 1)) },{{.*}} align 8
// CHECK: @_ZTV8Derived2 = linkonce_odr unnamed_addr constant { [5 x ptr], [3 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 0, i32 2)), ptr ptrauth (ptr @_ZN8Derived21cEv, i32 0, i64 [[DERIVED2_C_DISC:15537]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 0, i32 3)), ptr ptrauth (ptr @_ZN8Derived21eEv, i32 0, i64 209, ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 0, i32 4))], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 1, i32 2))] },{{.*}} align 8
// CHECK: @g_d2 = global { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 0, i32 2), i32 2, i64 [[BASE2_VTABLE_DISC]], ptr @g_d2), ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [5 x ptr], [3 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 1, i32 2), i32 2, i64 [[BASE1_VTABLE_DISC]], ptr getelementptr inbounds ({ ptr, ptr }, ptr @g_d2, i32 0, i32 1)) },{{.*}} align 8
// CHECK: @_ZTV8Derived3 = linkonce_odr unnamed_addr constant { [4 x ptr], [3 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV8Derived3, i32 0, i32 0, i32 2)), ptr ptrauth (ptr @_ZN8Derived31iEv, i32 0, i64 [[DERIVED3_I_DISC:19084]], ptr getelementptr inbounds ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV8Derived3, i32 0, i32 0, i32 3))], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC]], ptr getelementptr inbounds ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV8Derived3, i32 0, i32 1, i32 2))] },{{.*}} align 8
// CHECK: @g_d3 = global { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV8Derived3, i32 0, i32 0, i32 2), i32 2, i64 [[BASE1_VTABLE_DISC]], ptr @g_d3), ptr ptrauth (ptr getelementptr inbounds inrange(-16, 8) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV8Derived3, i32 0, i32 1, i32 2), i32 2, i64 [[BASE2_VTABLE_DISC]], ptr getelementptr inbounds ({ ptr, ptr }, ptr @g_d3, i32 0, i32 1)) },{{.*}} align 8
// CHECK: @g_vb1 = global %struct.VirtualBase1 zeroinitializer,{{.*}} align 8
// CHECK: @g_vb2 = global %struct.VirtualBase2 zeroinitializer,{{.*}} align 8
// CHECK: @g_d4 = global %struct.Derived4 zeroinitializer,{{.*}} align 8
// CHECK: @_ZTV12VirtualBase1 = linkonce_odr unnamed_addr constant { [6 x ptr] } { [6 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [6 x ptr] }, ptr @_ZTV12VirtualBase1, i32 0, i32 0, i32 4)), ptr ptrauth (ptr @_ZN12VirtualBase11fEv, i32 0, i64 [[VIRTUALBASE1_F_DISC:7987]], ptr getelementptr inbounds ({ [6 x ptr] }, ptr @_ZTV12VirtualBase1, i32 0, i32 0, i32 5))] },{{.*}} align 8
// CHECK: @_ZTT12VirtualBase1 = linkonce_odr unnamed_addr constant [2 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr] }, ptr @_ZTV12VirtualBase1, i32 0, i32 0, i32 4), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr] }, ptr @_ZTV12VirtualBase1, i32 0, i32 0, i32 4), i32 2)],{{.*}} align 8
// CHECK: @_ZTV12VirtualBase2 = linkonce_odr unnamed_addr constant { [5 x ptr], [4 x ptr] } { [5 x ptr] [ptr inttoptr (i64 8 to ptr), ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [4 x ptr] }, ptr @_ZTV12VirtualBase2, i32 0, i32 0, i32 3)), ptr ptrauth (ptr @_ZN12VirtualBase21gEv, i32 0, i64 [[VIRTUALBASE2_G_DISC:51224]], ptr getelementptr inbounds ({ [5 x ptr], [4 x ptr] }, ptr @_ZTV12VirtualBase2, i32 0, i32 0, i32 4))], [4 x ptr] [ptr null, ptr inttoptr (i64 -8 to ptr), ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [4 x ptr] }, ptr @_ZTV12VirtualBase2, i32 0, i32 1, i32 3))] },{{.*}} align 8
// CHECK: @_ZTT12VirtualBase2 = linkonce_odr unnamed_addr constant [2 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [4 x ptr] }, ptr @_ZTV12VirtualBase2, i32 0, i32 0, i32 3), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr] }, ptr @_ZTV12VirtualBase2, i32 0, i32 1, i32 3), i32 2)],{{.*}} align 8
// CHECK: @_ZTV8Derived4 = linkonce_odr unnamed_addr constant { [7 x ptr], [5 x ptr] } { [7 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 0, i32 4)), ptr ptrauth (ptr @_ZN12VirtualBase11fEv, i32 0, i64 [[VIRTUALBASE1_F_DISC]], ptr getelementptr inbounds ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 0, i32 5)), ptr ptrauth (ptr @_ZN8Derived41hEv, i32 0, i64 [[DERIVED4_H_DISC:31844]], ptr getelementptr inbounds ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 0, i32 6))], [5 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr inttoptr (i64 -8 to ptr), ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC]], ptr getelementptr inbounds ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 1, i32 3)), ptr ptrauth (ptr @_ZN12VirtualBase21gEv, i32 0, i64 [[VIRTUALBASE2_G_DISC]], ptr getelementptr inbounds ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 1, i32 4))] },{{.*}} align 8
// CHECK: @_ZTT8Derived4 = linkonce_odr unnamed_addr constant [7 x ptr] [ptr ptrauth (ptr getelementptr inbounds inrange(-32, 24) ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 0, i32 4), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr] }, ptr @_ZTC8Derived40_12VirtualBase1, i32 0, i32 0, i32 4), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-32, 16) ({ [6 x ptr] }, ptr @_ZTC8Derived40_12VirtualBase1, i32 0, i32 0, i32 4), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-24, 16) ({ [5 x ptr], [4 x ptr] }, ptr @_ZTC8Derived48_12VirtualBase2, i32 0, i32 0, i32 3), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-24, 8) ({ [5 x ptr], [4 x ptr] }, ptr @_ZTC8Derived48_12VirtualBase2, i32 0, i32 1, i32 3), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-32, 24) ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 0, i32 4), i32 2), ptr ptrauth (ptr getelementptr inbounds inrange(-24, 16) ({ [7 x ptr], [5 x ptr] }, ptr @_ZTV8Derived4, i32 0, i32 1, i32 3), i32 2)],{{.*}} align 8
// CHECK: @_ZTC8Derived40_12VirtualBase1 = linkonce_odr unnamed_addr constant { [6 x ptr] } { [6 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [6 x ptr] }, ptr @_ZTC8Derived40_12VirtualBase1, i32 0, i32 0, i32 4)), ptr ptrauth (ptr @_ZN12VirtualBase11fEv, i32 0, i64 [[VIRTUALBASE1_F_DISC]], ptr getelementptr inbounds ({ [6 x ptr] }, ptr @_ZTC8Derived40_12VirtualBase1, i32 0, i32 0, i32 5))] },{{.*}} align 8
// CHECK: @_ZTC8Derived48_12VirtualBase2 = linkonce_odr unnamed_addr constant { [5 x ptr], [4 x ptr] } { [5 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr null, ptr null, ptr ptrauth (ptr @_ZN5Base21bEv, i32 0, i64 [[BASE2_B_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [4 x ptr] }, ptr @_ZTC8Derived48_12VirtualBase2, i32 0, i32 0, i32 3)), ptr ptrauth (ptr @_ZN12VirtualBase21gEv, i32 0, i64 [[VIRTUALBASE2_G_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [4 x ptr] }, ptr @_ZTC8Derived48_12VirtualBase2, i32 0, i32 0, i32 4))], [4 x ptr] [ptr null, ptr inttoptr (i64 8 to ptr), ptr null, ptr ptrauth (ptr @_ZN5Base11aEv, i32 0, i64 [[BASE1_A_DISC]], ptr getelementptr inbounds ({ [5 x ptr], [4 x ptr] }, ptr @_ZTC8Derived48_12VirtualBase2, i32 0, i32 1, i32 3))] },{{.*}} align 8

struct Base1 { virtual void a() {} };
struct Base2 { virtual void b() {} };
struct Derived1 : public Base1, public Base2 {
  virtual void c() {}
  virtual void d() {}
};
struct Derived2 : public Base2, public Base1 {
  virtual void c() {}
  virtual void e() {}
};

struct Derived3 : public Base1, public Base2 {
  constexpr Derived3(){}
  virtual void i() {}
};

Base1 g_b1;
Base2 g_b2;
Derived1 g_d1;
Derived2 g_d2;
Derived3 g_d3;

extern "C" void test_basic_inheritance() {
  Base1 g_b1;
  Base2 g_b2;
  Derived1 g_d1;
  Derived2 g_d2;
  Derived3 g_d3;
}

struct VirtualBase1 : virtual Base1 {
  VirtualBase1(){}
  virtual void f() {}
};
struct VirtualBase2 : virtual Base1, Base2 {
  VirtualBase2(){}
  virtual void g() {}
};
struct Derived4 : VirtualBase1, VirtualBase2 {
  virtual void h() {}
};
struct Derived5 : VirtualBase2, VirtualBase1 {
  virtual void h() {}
};

// DARWIN-LABEL: define {{.*}} ptr @_ZN12VirtualBase1C1Ev
// ELF-LABEL:    define {{.*}} void @_ZN12VirtualBase1C1Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN12VirtualBase2C1Ev
// ELF-LABEL:    define {{.*}} void @_ZN12VirtualBase2C1Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN8Derived4C1Ev
// ELF-LABEL:    define {{.*}} void @_ZN8Derived4C1Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN8Derived5C1Ev
// ELF-LABEL:    define {{.*}} void @_ZN8Derived5C1Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])


VirtualBase1 g_vb1;
VirtualBase2 g_vb2;
Derived4 g_d4;
Derived5 g_d5;


extern "C" void cross_check_vtables(Base1 *b1,
                   Base2 *b2,
                   Derived1 *d1,
                   Derived2 *d2,
                   Derived3 *d3,
                   VirtualBase1 *vb1,
                   VirtualBase2 *vb2,
                   Derived4 *d4,
                   Derived4 *d5) {
  asm("; b1->a()" ::: "memory");
  b1->a();
  asm("; b2->b()" ::: "memory");
  b2->b();
  asm("; d1->a()" ::: "memory");
  d1->a();
  asm("; d1->c()" ::: "memory");
  d1->c();
  asm("; d2->a()" ::: "memory");
  d2->a();
  asm("; d2->c()" ::: "memory");
  d2->c();
  asm("; d3->a()" ::: "memory");
  d3->a();
  asm("; d3->b()" ::: "memory");
  d3->b();
  asm("; d3->i()" ::: "memory");
  d3->i();
  asm("; vb1->a()" ::: "memory");
  vb1->a();
  asm("; vb1->f()" ::: "memory");
  vb1->f();
  asm("; vb2->a()" ::: "memory");
  vb2->a();
  asm("; vb2->g()" ::: "memory");
  vb2->g();
  asm("; d4->a()" ::: "memory");
  d4->a();
  asm("; d4->b()" ::: "memory");
  d4->b();
  asm("; d4->f()" ::: "memory");
  d4->f();
  asm("; d4->g()" ::: "memory");
  d4->g();
  asm("; d4->h()" ::: "memory");
  d4->h();
  asm("; d5->a()" ::: "memory");
  d5->a();
  asm("; d5->b()" ::: "memory");
  d5->b();
  asm("; d5->f()" ::: "memory");
  d5->f();
  asm("; d5->g()" ::: "memory");
  d5->g();
  asm("; d5->h()" ::: "memory");
  d5->h();
}

// CHECK-LABEL: define{{.*}} void @cross_check_vtables(
// CHECK: "; b1->a()",
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; b2->b()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_B_DISC]])
// CHECK: "; d1->a()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; d1->c()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[DERIVED1_C_DISC]])
// CHECK: "; d2->a()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; d2->c()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[DERIVED2_C_DISC]])
// CHECK: "; d3->a()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; d3->b()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_B_DISC]])
// CHECK: "; d3->i()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[DERIVED3_I_DISC]])
// CHECK: "; vb1->a()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; vb1->f()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[VIRTUALBASE1_F_DISC]])
// CHECK: "; vb2->a()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; vb2->g()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[VIRTUALBASE2_G_DISC]])
// CHECK: "; d4->a()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_A_DISC]])
// CHECK: "; d4->b()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_B_DISC]])
// CHECK: "; d4->f()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[VIRTUALBASE1_F_DISC]])
// CHECK: "; d4->g()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[VIRTUALBASE2_G_DISC]])
// CHECK: "; d4->h()"
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[DERIVED4_H_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN5Base1C2Ev
// ELF-LABEL:    define {{.*}} void @_ZN5Base1C2Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN5Base2C2Ev
// ELF-LABEL:    define {{.*}} void @_ZN5Base2C2Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN8Derived1C2Ev
// ELF-LABEL:    define {{.*}} void @_ZN8Derived1C2Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN8Derived2C2Ev
// ELF-LABEL:    define {{.*}} void @_ZN8Derived2C2Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])

// DARWIN-LABEL: define {{.*}} ptr @_ZN8Derived3C2Ev
// ELF-LABEL:    define {{.*}} void @_ZN8Derived3C2Ev
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE1_VTABLE_DISC]])
// CHECK: call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 [[BASE2_VTABLE_DISC]])
