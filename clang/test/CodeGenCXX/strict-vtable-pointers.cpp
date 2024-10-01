// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -fstrict-vtable-pointers -std=c++11 -disable-llvm-passes -O2 -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=CHECK-CTORS %s < %t.ll
// RUN: FileCheck --check-prefix=CHECK-NEW %s < %t.ll
// RUN: FileCheck --check-prefix=CHECK-DTORS %s < %t.ll
// RUN: FileCheck --check-prefix=CHECK-LINK-REQ %s < %t.ll

typedef __typeof__(sizeof(0)) size_t;
void *operator new(size_t, void *) throw();
using uintptr_t = unsigned long long;

struct NotTrivialDtor {
  ~NotTrivialDtor();
};

struct DynamicBase1 {
  NotTrivialDtor obj;
  virtual void foo();
};

struct DynamicDerived : DynamicBase1 {
  void foo() override;
};

struct DynamicBase2 {
  virtual void bar();
  ~DynamicBase2() {
    bar();
  }
};

struct DynamicDerivedMultiple : DynamicBase1, DynamicBase2 {
  void foo() override;
  void bar() override;
};

struct StaticBase {
  NotTrivialDtor obj;
  void bar();
};

struct DynamicFromStatic : StaticBase {
  virtual void bar();
};

struct DynamicFromVirtualStatic1 : virtual StaticBase {
};

struct DynamicFromVirtualStatic2 : virtual StaticBase {
};

struct DynamicFrom2Virtuals : DynamicFromVirtualStatic1,
                              DynamicFromVirtualStatic2 {
};

// CHECK-NEW-LABEL: define{{.*}} void @_Z12LocalObjectsv()
// CHECK-NEW-NOT: @llvm.launder.invariant.group.p0(
// CHECK-NEW-LABEL: {{^}}}
void LocalObjects() {
  DynamicBase1 DB;
  DB.foo();
  DynamicDerived DD;
  DD.foo();

  DynamicBase2 DB2;
  DB2.bar();

  StaticBase SB;
  SB.bar();

  DynamicDerivedMultiple DDM;
  DDM.foo();
  DDM.bar();

  DynamicFromStatic DFS;
  DFS.bar();
  DynamicFromVirtualStatic1 DFVS1;
  DFVS1.bar();
  DynamicFrom2Virtuals DF2V;
  DF2V.bar();
}

struct DynamicFromVirtualStatic1;
// CHECK-CTORS-LABEL: define linkonce_odr void @_ZN25DynamicFromVirtualStatic1C1Ev
// CHECK-CTORS-NOT: @llvm.launder.invariant.group.p0(
// CHECK-CTORS-LABEL: {{^}}}

struct DynamicFrom2Virtuals;
// CHECK-CTORS-LABEL: define linkonce_odr void @_ZN20DynamicFrom2VirtualsC1Ev
// CHECK-CTORS: call ptr @llvm.launder.invariant.group.p0(
// CHECK-CTORS-LABEL: {{^}}}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9Pointers1v()
// CHECK-NEW-NOT: @llvm.launder.invariant.group.p0(
// CHECK-NEW-LABEL: call void @_ZN12DynamicBase1C1Ev(

// CHECK-NEW: %[[THIS3:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[THIS2:.*]])
// CHECK-NEW: call void @_ZN14DynamicDerivedC1Ev(ptr {{[^,]*}} %[[THIS3]])
// CHECK-NEW-LABEL: {{^}}}
void Pointers1() {
  DynamicBase1 *DB = new DynamicBase1;
  DB->foo();

  DynamicDerived *DD = new (DB) DynamicDerived;
  DD->foo();
  DD->~DynamicDerived();
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z14HackingObjectsv()
// CHECK-NEW:  call void @_ZN12DynamicBase1C1Ev
// CHECK-NEW:  call ptr @llvm.launder.invariant.group.p0(
// CHECK-NEW:  call void @_ZN14DynamicDerivedC1Ev(
// CHECK-NEW:  call ptr @llvm.launder.invariant.group.p0(
// CHECK-NEW: call void @_ZN12DynamicBase1C1Ev(
// CHECK-NEW-LABEL: {{^}}}
void HackingObjects() {
  DynamicBase1 DB;
  DB.foo();

  DynamicDerived *DB2 = new (&DB) DynamicDerived;
  // Using DB now is prohibited.
  DB2->foo();
  DB2->~DynamicDerived();

  // We have to get back to the previous type to avoid calling wrong destructor
  new (&DB) DynamicBase1;
  DB.foo();
}

/*** Testing Constructors ***/
struct DynamicBase1;
// CHECK-CTORS-LABEL: define linkonce_odr void @_ZN12DynamicBase1C2Ev(
// CHECK-CTORS-NOT: call ptr @llvm.launder.invariant.group.p0(
// CHECK-CTORS-LABEL: {{^}}}

struct DynamicDerived;

// CHECK-CTORS-LABEL: define linkonce_odr void @_ZN14DynamicDerivedC2Ev(
// CHECK-CTORS: %[[THIS0:.*]] = load ptr, ptr {{.*}}
// CHECK-CTORS: %[[THIS2:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[THIS1:.*]])
// CHECK-CTORS: call void @_ZN12DynamicBase1C2Ev(ptr {{[^,]*}} %[[THIS2]])

// CHECK-CTORS: store {{.*}} %[[THIS0]]
// CHECK-CTORS-LABEL: {{^}}}

struct DynamicDerivedMultiple;
// CHECK-CTORS-LABEL: define linkonce_odr void @_ZN22DynamicDerivedMultipleC2Ev(

// CHECK-CTORS: %[[THIS0:.*]] = load ptr, ptr {{.*}}
// CHECK-CTORS: %[[THIS2:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[THIS0]])
// CHECK-CTORS: call void @_ZN12DynamicBase1C2Ev(ptr {{[^,]*}} %[[THIS2]])

// CHECK-CTORS: call ptr @llvm.launder.invariant.group.p0(

// CHECK-CTORS: call void @_ZN12DynamicBase2C2Ev(
// CHECK-CTORS-NOT: @llvm.launder.invariant.group.p0

// CHECK-CTORS: store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV22DynamicDerivedMultiple, i32 0, i32 0, i32 2), ptr %[[THIS0]]
// CHECK-CTORS: %[[THIS_ADD:.*]] = getelementptr inbounds i8, ptr %[[THIS0]], i64 16

// CHECK-CTORS: store ptr getelementptr inbounds inrange(-16, 8) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV22DynamicDerivedMultiple, i32 0, i32 1, i32 2), ptr %[[THIS_ADD]]
// CHECK-CTORS-LABEL: {{^}}}

struct DynamicFromStatic;
// CHECK-CTORS-LABEL: define linkonce_odr void @_ZN17DynamicFromStaticC2Ev(
// CHECK-CTORS-NOT: @llvm.launder.invariant.group.p0(
// CHECK-CTORS-LABEL: {{^}}}

struct A {
  virtual void foo();
  int m;
};
struct B : A {
  void foo() override;
};

union U {
  A a;
  B b;
};

void changeToB(U *u);
void changeToA(U *u);

void g2(A *a) {
  a->foo();
}
// We have to guard access to union fields with invariant.group, because
// it is very easy to skip the barrier with unions. In this example the inlined
// g2 will produce loads with the same !invariant.group metadata, and
// u->a and u->b would use the same pointer.
// CHECK-NEW-LABEL: define{{.*}} void @_Z14UnionsBarriersP1U
void UnionsBarriers(U *u) {
  // CHECK-NEW: call void @_Z9changeToBP1U(
  changeToB(u);
  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: call void @_Z2g2P1A(ptr
  g2(&u->b);
  // CHECK-NEW: call void @_Z9changeToAP1U(ptr
  changeToA(u);
  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  // call void @_Z2g2P1A(ptr %a)
  g2(&u->a);
  // CHECK-NEW-NOT: call ptr @llvm.launder.invariant.group.p0(ptr
}

struct HoldingVirtuals {
  A a;
};

struct Empty {};
struct AnotherEmpty {
  Empty e;
};
union NoVptrs {
  int a;
  AnotherEmpty empty;
};
void take(AnotherEmpty &);

// CHECK-NEW-LABEL: noBarriers
void noBarriers(NoVptrs &noVptrs) {
  // CHECK-NEW-NOT: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: 42
  noVptrs.a += 42;
  // CHECK-NEW-NOT: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: call void @_Z4takeR12AnotherEmpty(
  take(noVptrs.empty);
}

union U2 {
  HoldingVirtuals h;
  int z;
};
void take(HoldingVirtuals &);

// CHECK-NEW-LABEL: define{{.*}} void @_Z15UnionsBarriers2R2U2
void UnionsBarriers2(U2 &u) {
  // CHECK-NEW-NOT: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: 42
  u.z += 42;
  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: call void @_Z4takeR15HoldingVirtuals(
  take(u.h);
}

struct VirtualInBase : HoldingVirtuals, Empty {
};

struct VirtualInVBase : virtual Empty, virtual HoldingVirtuals {
};

// It has vtable by virtual inheritance.
struct VirtualInheritance : virtual Empty {
};

union U3 {
  VirtualInBase v1;
  VirtualInBase v2;
  VirtualInheritance v3;
  int z;
};

void take(VirtualInBase &);
void take(VirtualInVBase &);
void take(VirtualInheritance &);

void UnionsBarrier3(U3 &u) {
  // CHECK-NEW-NOT: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: 42
  u.z += 42;
  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: call void @_Z4takeR13VirtualInBase(
  take(u.v1);
  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: call void @_Z4takeR13VirtualInBase(
  take(u.v2);

  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  // CHECK-NEW: call void @_Z4takeR18VirtualInheritance(
  take(u.v3);
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z7comparev()
void compare() {
  A *a = new A;
  a->foo();
  // CHECK-NEW: call ptr @llvm.launder.invariant.group.p0(ptr
  A *b = new (a) B;

  // CHECK-NEW: %[[a:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr
  // CHECK-NEW: %[[b:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr
  // CHECK-NEW: %cmp = icmp eq ptr %[[a]], %[[b]]
  if (a == b)
    b->foo();
}

// CHECK-NEW-LABEL: compare2
bool compare2(A *a, A *a2) {
  // CHECK-NEW: %[[a:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr
  // CHECK-NEW: %[[b:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr
  // CHECK-NEW: %cmp = icmp ult ptr %[[a]], %[[b]]
  return a < a2;
}
// CHECK-NEW-LABEL: compareIntPointers
bool compareIntPointers(int *a, int *b) {
  // CHECK-NEW-NOT: call ptr @llvm.strip.invariant.group
  return a == b;
}

struct HoldingOtherVirtuals {
  B b;
};

// There is no need to add barriers for comparision of pointer to classes
// that are not dynamic.
// CHECK-NEW-LABEL: compare5
bool compare5(HoldingOtherVirtuals *a, HoldingOtherVirtuals *b) {
  // CHECK-NEW-NOT: call ptr @llvm.strip.invariant.group
  return a == b;
}
// CHECK-NEW-LABEL: compareNull
bool compareNull(A *a) {
  // CHECK-NEW-NOT: call ptr @llvm.strip.invariant.group

  if (a != nullptr)
    return false;
  if (!a)
    return false;
  return a == nullptr;
}

struct X;
// We have to also introduce the barriers if comparing pointers to incomplete
// objects
// CHECK-NEW-LABEL: define{{.*}} zeroext i1 @_Z8compare4P1XS0_
bool compare4(X *x, X *x2) {
  // CHECK-NEW: %[[x:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr
  // CHECK-NEW: %[[x2:.*]] = call ptr @llvm.strip.invariant.group.p0(ptr
  // CHECK-NEW: %cmp = icmp eq ptr %[[x]], %[[x2]]
  return x == x2;
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z7member1P20HoldingOtherVirtuals(
void member1(HoldingOtherVirtuals *p) {

  // CHECK-NEW-NOT: call ptr @llvm.strip.invariant.group.p0(
  (void)p->b;
}

// CHECK-NEW-LABEL: member2
void member2(A *a) {
  // CHECK-NEW: call ptr @llvm.strip.invariant.group.p0
  (void)a->m;
}

// Check if from comparison of addresses of member we can't infer the equality
// of ap and bp.
// CHECK-NEW-LABEL: @_Z18testCompareMembersv(
void testCompareMembers() {
  // CHECK-NEW:    [[AP:%.*]] = alloca ptr
  // CHECK-NEW:    [[APM:%.*]] = alloca ptr
  // CHECK-NEW:    [[BP:%.*]] = alloca ptr
  // CHECK-NEW:    [[BPM:%.*]] = alloca ptr

  A *ap = new A;
  // CHECK-NEW:   call void %{{.*}}(ptr {{[^,]*}} %{{.*}})
  ap->foo();
  // CHECK-NEW:    [[TMP7:%.*]] = load ptr, ptr [[AP]]
  // CHECK-NEW:    [[TMP9:%.*]] = call ptr @llvm.strip.invariant.group.p0(ptr [[TMP7]])
  // CHECK-NEW:    [[M:%.*]] = getelementptr inbounds nuw [[STRUCT_A:%.*]], ptr [[TMP9]], i32 0, i32 1
  // CHECK-NEW:    store ptr [[M]], ptr [[APM]]
  int *const apm = &ap->m;

  B *bp = new (ap) B;

  // CHECK-NEW:    [[TMP20:%.*]] = load ptr, ptr [[BP]]
  // CHECK-NEW:    [[TMP23:%.*]] = call ptr @llvm.strip.invariant.group.p0(ptr [[TMP20]])
  // CHECK-NEW:    [[M4:%.*]] = getelementptr inbounds nuw [[STRUCT_A]], ptr [[TMP23]], i32 0, i32 1
  // CHECK-NEW:    store ptr [[M4]], ptr [[BPM]]
  int *const bpm = &bp->m;

  // CHECK-NEW:    [[TMP25:%.*]] = load ptr, ptr [[APM]]
  // CHECK-NEW:    [[TMP26:%.*]] = load ptr, ptr [[BPM]]
  // CHECK-NEW-NOT: strip.invariant.group
  // CHECK-NEW-NOT: launder.invariant.group
  // CHECK-NEW:    [[CMP:%.*]] = icmp eq ptr [[TMP25]], [[TMP26]]
  if (apm == bpm) {
    bp->foo();
  }
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast1P1A(ptr
void testCast1(A *a) {
  // Here we get rid of dynamic info
  // CHECK-NEW: call ptr @llvm.strip.invariant.group
  auto *v = (void *)a;

  // CHECK-NEW: call ptr @llvm.strip.invariant.group
  auto i2 = (uintptr_t)a;
  (void)i2;

  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  // CHECK-NEW-NOT: @llvm.launder.invariant.group

  // The information is already stripped
  auto i = (uintptr_t)v;
}

struct Incomplete;
// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast2P10Incomplete(ptr
void testCast2(Incomplete *I) {
  // Here we get rid of potential dynamic info
  // CHECK-NEW: call ptr @llvm.strip.invariant.group
  auto *v = (void *)I;

  // CHECK-NEW: call ptr @llvm.strip.invariant.group
  auto i2 = (uintptr_t)I;
  (void)i2;

  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  // CHECK-NEW-NOT: @llvm.launder.invariant.group

  // The information is already stripped
  auto i = (uintptr_t)v;
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast3y(
void testCast3(uintptr_t i) {
  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  // CHECK-NEW: @llvm.launder.invariant.group
  A *a3 = (A *)i;
  (void)a3;

  auto *v2 = (void *)i;

  // CHECK-NEW: @llvm.launder.invariant.group
  A *a2 = (A *)v2;
  (void)a2;

  // CHECK-NEW-NOT: @llvm.launder.invariant.group
  auto *v3 = (void *)i;
  (void)v3;
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast4y(
void testCast4(uintptr_t i) {
  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  // CHECK-NEW: @llvm.launder.invariant.group
  auto *a3 = (Incomplete *)i;
  (void)a3;

  // CHECK-NEW: @llvm.launder.invariant.group
  auto *v2 = (void *)i;
  // CHECK-NEW-NOT: @llvm.launder.invariant.group
  auto *a2 = (Incomplete *)v2;
  (void)a2;
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast5P1B(
void testCast5(B *b) {
  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  // CHECK-NEW-NOT: @llvm.launder.invariant.group
  A *a = b;
  (void)a;

  auto *b2 = (B *)a;
  (void)b2;
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast6P1A(
void testCast6(A *a) {

  // CHECK-NEW: @llvm.strip.invariant.group
  auto *I = (Incomplete *)a;
  (void)I;
  // CHECK-NEW: @llvm.launder.invariant.group
  auto *a2 = (A *)I;
  (void)a2;

  // CHECK-NEW: @llvm.strip.invariant.group
  auto *E = (Empty *)a;
  (void)E;

  // CHECK-NEW: @llvm.launder.invariant.group
  auto *a3 = (A *)E;
  (void)a3;

  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  auto i = (uintptr_t)E;
  (void)i;
}

class Incomplete2;
// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast7P10Incomplete(
void testCast7(Incomplete *I) {
  // CHECK-NEW-NOT: @llvm.strip.invariant.group

  // Incomplete2 could be dynamic where Incomplete may not be dynamic, thus
  // launder is needed.  We don't strip firstly because launder is sufficient.

  // CHECK-NEW: @llvm.launder.invariant.group
  auto *I2 = (Incomplete2 *)I;
  (void)I2;
  // CHECK-NEW-LABEL: ret void
}

template <typename Base>
struct PossiblyDerivingFromDynamicBase : Base {
};

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast8P10Incomplete(
void testCast8(Incomplete *I) {
  // CHECK-NEW-NOT: @llvm.strip.invariant.group
  // CHECK-NEW: @llvm.launder.invariant.group
  auto *P = (PossiblyDerivingFromDynamicBase<Incomplete> *)I;
  (void)P;

  // CHECK-NEW: @llvm.launder.invariant.group
  auto *P2 = (PossiblyDerivingFromDynamicBase<Empty> *)I;
  (void)P2;

  // CHECK-NEW: @llvm.launder.invariant.group
  auto *P3 = (PossiblyDerivingFromDynamicBase<A> *)I;
  (void)P3;

  // CHECK-NEW-NOT: @llvm.launder.invariant.group
  auto *a3 = (A *)P3;

  // CHECK-NEW-LABEL: ret void
}

// CHECK-NEW-LABEL: define{{.*}} void @_Z9testCast9
void testCast9(PossiblyDerivingFromDynamicBase<Incomplete> *P) {
  // CHECK-NEW: @llvm.strip.invariant.group
  auto *V = (void *)P;

  // CHECK-NEW-LABEL: ret void
}

/** DTORS **/
// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN10StaticBaseD2Ev(
// CHECK-DTORS-NOT: call ptr @llvm.launder.invariant.group.p0(
// CHECK-DTORS-LABEL: {{^}}}

// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN25DynamicFromVirtualStatic2D2Ev(
// CHECK-DTORS-NOT: invariant.barrier
// CHECK-DTORS-LABEL: {{^}}}

// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN17DynamicFromStaticD2Ev
// CHECK-DTORS-NOT: call ptr @llvm.launder.invariant.group.p0(
// CHECK-DTORS-LABEL: {{^}}}

// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN22DynamicDerivedMultipleD2Ev(

// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN12DynamicBase2D2Ev(
// CHECK-DTORS: call ptr @llvm.launder.invariant.group.p0(
// CHECK-DTORS-LABEL: {{^}}}

// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN12DynamicBase1D2Ev
// CHECK-DTORS: call ptr @llvm.launder.invariant.group.p0(
// CHECK-DTORS-LABEL: {{^}}}

// CHECK-DTORS-LABEL: define linkonce_odr void @_ZN14DynamicDerivedD2Ev
// CHECK-DTORS-NOT: call ptr @llvm.launder.invariant.group.p0(
// CHECK-DTORS-LABEL: {{^}}}

// CHECK-LINK-REQ: !llvm.module.flags = !{![[FIRST:[0-9]+]], ![[SEC:[0-9]+]]{{.*}}}

// CHECK-LINK-REQ: ![[FIRST]] = !{i32 1, !"StrictVTablePointers", i32 1}
// CHECK-LINK-REQ: ![[SEC]] = !{i32 3, !"StrictVTablePointersRequirement", ![[META:.*]]}
// CHECK-LINK-REQ: ![[META]] = !{!"StrictVTablePointers", i32 1}
