// RUN: %clang_cc1 -std=c++11 -fobjc-arc -fblocks -triple x86_64-apple-darwin10.0.0 -fobjc-runtime-has-weak -emit-llvm -o - %s | FileCheck %s

struct ObjCMember {
  id member;
};

struct ObjCArrayMember {
  id member[2][3];
};

struct ObjCBlockMember {
  int (^bp)(int);
};

// CHECK: %[[STRUCT_CONTAINSWEAK:.*]] = type { %[[STRUCT_WEAK:.*]] }
// CHECK: %[[STRUCT_WEAK]] = type { i8* }

// The Weak object that is passed is destructed in this constructor.

// CHECK: define{{.*}} void @_ZN12ContainsWeakC2E4Weak(
// CHECK: call void @_ZN4WeakC1ERKS_(
// CHECK: call void @_ZN4WeakD1Ev(

// Check that the Weak object passed to this constructor is not destructed after
// the delegate constructor is called.

// CHECK: define{{.*}} void @_ZN12ContainsWeakC1E4Weak(
// CHECK: call void @_ZN12ContainsWeakC2E4Weak(
// CHECK-NEXT: ret void

struct Weak {
  Weak(id);
  __weak id x;
};

struct ContainsWeak {
  ContainsWeak(Weak);
  Weak w;
};

ContainsWeak::ContainsWeak(Weak a) : w(a) {}

// The Weak object that is passed is destructed in this constructor.

// CHECK: define{{.*}} void @_ZN4BaseC2E4Weak(
// CHECK: call void @_ZN4WeakD1Ev(
// CHECK: ret void

// Check that the Weak object passed to this constructor is not destructed after
// the delegate constructor is called.

// CHECK: define linkonce_odr void @_ZN7DerivedCI14BaseE4Weak(
// CHECK: call void @_ZN7DerivedCI24BaseE4Weak(
// CHECK-NEXT: ret void

struct Base {
  Base(Weak);
};

Base::Base(Weak a) {}

struct Derived : Base {
  using Base::Base;
};

Derived d(Weak(0));

// CHECK-LABEL: define{{.*}} void @_Z42test_ObjCMember_default_construct_destructv(
void test_ObjCMember_default_construct_destruct() {
  // CHECK: call void @_ZN10ObjCMemberC1Ev
  // CHECK: call void @_ZN10ObjCMemberD1Ev
  ObjCMember m1;
}

// CHECK-LABEL: define{{.*}} void @_Z39test_ObjCMember_copy_construct_destruct10ObjCMember
void test_ObjCMember_copy_construct_destruct(ObjCMember m1) {
  // CHECK: call void @_ZN10ObjCMemberC1ERKS_
  // CHECK: call void @_ZN10ObjCMemberD1Ev
  ObjCMember m2 = m1;
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z27test_ObjCMember_copy_assign10ObjCMemberS_
void test_ObjCMember_copy_assign(ObjCMember m1, ObjCMember m2) {
  // CHECK: {{call.*_ZN10ObjCMemberaSERKS_}}
  m1 = m2;
  // CHECK-NEXT: call void @_ZN10ObjCMemberD1Ev(
  // CHECK-NEXT: call void @_ZN10ObjCMemberD1Ev(
  // CHECK-NEXT: ret void
}

// Implicitly-generated copy assignment operator for ObjCMember
// CHECK:    {{define linkonce_odr.*@_ZN10ObjCMemberaSERKS_}}
// CHECK:      call void @llvm.objc.storeStrong
// CHECK:      ret

// CHECK-LABEL: define{{.*}} void @_Z47test_ObjCArrayMember_default_construct_destructv
void test_ObjCArrayMember_default_construct_destruct() {
  // CHECK: call void @_ZN15ObjCArrayMemberC1Ev
  ObjCArrayMember m1;
  // CHECK: call void @_ZN15ObjCArrayMemberD1Ev
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z44test_ObjCArrayMember_copy_construct_destruct15ObjCArrayMember
void test_ObjCArrayMember_copy_construct_destruct(ObjCArrayMember m1) {
  // CHECK: call void @_ZN15ObjCArrayMemberC1ERKS_
  ObjCArrayMember m2 = m1;
  // CHECK: call void @_ZN15ObjCArrayMemberD1Ev
  // CHECK: ret void
}

void test_ObjCArrayMember_copy_assign(ObjCArrayMember m1, ObjCArrayMember m2) {
  // CHECK: {{call.*@_ZN15ObjCArrayMemberaSERKS_}}
  m1 = m2;
  // CHECK-NEXT: call void @_ZN15ObjCArrayMemberD1Ev(
  // CHECK-NEXT: call void @_ZN15ObjCArrayMemberD1Ev(
  // CHECK-NEXT: ret void
}

// Implicitly-generated copy assignment operator for ObjCArrayMember
// CHECK: {{define linkonce_odr.*@_ZN15ObjCArrayMemberaSERKS_}}
// CHECK:      call void @llvm.objc.storeStrong
// CHECK-NEXT: br label
// CHECK: ret

// CHECK-LABEL: define{{.*}} void @_Z47test_ObjCBlockMember_default_construct_destructv
void test_ObjCBlockMember_default_construct_destruct() {
  // CHECK: call void @_ZN15ObjCBlockMemberC1Ev
  ObjCBlockMember m;
  // CHECK-NEXT: call void @_ZN15ObjCBlockMemberD1Ev
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z44test_ObjCBlockMember_copy_construct_destruct15ObjCBlockMember
void test_ObjCBlockMember_copy_construct_destruct(ObjCBlockMember m1) {
  // CHECK: call void @_ZN15ObjCBlockMemberC1ERKS_
  ObjCBlockMember m2 = m1;
  // CHECK-NEXT: call void @_ZN15ObjCBlockMemberD1Ev(
  // CHECK-NEXT: call void @_ZN15ObjCBlockMemberD1Ev(
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z32test_ObjCBlockMember_copy_assign15ObjCBlockMemberS_
void test_ObjCBlockMember_copy_assign(ObjCBlockMember m1, ObjCBlockMember m2) {
  // CHECK: {{call.*_ZN15ObjCBlockMemberaSERKS_}}
  m1 = m2;
  // CHECK-NEXT: call void @_ZN15ObjCBlockMemberD1Ev(
  // CHECK-NEXT: call void @_ZN15ObjCBlockMemberD1Ev(
  // CHECK-NEXT: ret void
}

// Implicitly-generated copy assignment operator for ObjCBlockMember
// CHECK:    define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) {{%.*}}* @_ZN15ObjCBlockMemberaSERKS_(
// CHECK:      [[T0:%.*]] = getelementptr inbounds [[T:%.*]], [[T:%.*]]* {{%.*}}, i32 0, i32 0
// CHECK-NEXT: [[T1:%.*]] = load i32 (i32)*, i32 (i32)** [[T0]], align 8
// CHECK-NEXT: [[T2:%.*]] = bitcast i32 (i32)* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = call i8* @llvm.objc.retainBlock(i8* [[T2]])
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i32 (i32)*
// CHECK-NEXT: [[T5:%.*]] = getelementptr inbounds [[T]], [[T]]* {{%.*}}, i32 0, i32 0
// CHECK-NEXT: [[T6:%.*]] = load i32 (i32)*, i32 (i32)** [[T5]], align 8
// CHECK-NEXT: store i32 (i32)* [[T4]], i32 (i32)** [[T5]]
// CHECK-NEXT: [[T7:%.*]] = bitcast i32 (i32)* [[T6]] to i8*
// CHECK-NEXT: call void @llvm.objc.release(i8* [[T7]])
// CHECK-NEXT: ret

// Check that the Weak object passed to this constructor is not destructed after
// the delegate constructor is called.

// CHECK: define linkonce_odr void @_ZN7DerivedCI24BaseE4Weak(
// CHECK: call void @_ZN4BaseC2E4Weak(
// CHECK-NEXT: ret void

// Implicitly-generated default constructor for ObjCMember
// CHECK-LABEL: define linkonce_odr void @_ZN10ObjCMemberC2Ev
// CHECK-NOT: objc_release
// CHECK: store i8* null
// CHECK-NEXT: ret void

// Implicitly-generated destructor for ObjCMember
// CHECK-LABEL: define linkonce_odr void @_ZN10ObjCMemberD2Ev
// CHECK: call void @llvm.objc.storeStrong
// CHECK: ret void

// Implicitly-generated copy constructor for ObjCMember
// CHECK-LABEL: define linkonce_odr void @_ZN10ObjCMemberC2ERKS_
// CHECK-NOT: objc_release
// CHECK: call i8* @llvm.objc.retain
// CHECK-NEXT: store i8*
// CHECK-NEXT: ret void

// Implicitly-generated default constructor for ObjCArrayMember
// CHECK-LABEL: define linkonce_odr void @_ZN15ObjCArrayMemberC2Ev
// CHECK: call void @llvm.memset.p0i8.i64
// CHECK: ret

// Implicitly-generated destructor for ObjCArrayMember
// CHECK-LABEL:    define linkonce_odr void @_ZN15ObjCArrayMemberD2Ev
// CHECK:      [[BEGIN:%.*]] = getelementptr inbounds [2 x [3 x i8*]], [2 x [3 x i8*]]*
// CHECK-NEXT: [[END:%.*]] = getelementptr inbounds i8*, i8** [[BEGIN]], i64 6
// CHECK-NEXT: br label
// CHECK:      [[PAST:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
// CHECK-NEXT: [[CUR]] = getelementptr inbounds i8*, i8** [[PAST]], i64 -1
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[CUR]], i8* null)
// CHECK-NEXT: [[T1:%.*]] = icmp eq i8** [[CUR]], [[BEGIN]]
// CHECK-NEXT: br i1 [[T1]],
// CHECK:      ret void

// Implicitly-generated copy constructor for ObjCArrayMember
// CHECK-LABEL: define linkonce_odr void @_ZN15ObjCArrayMemberC2ERKS_
// CHECK: call i8* @llvm.objc.retain
// CHECK-NEXT: store i8*
// CHECK: br i1
// CHECK: ret

// Implicitly-generated default constructor for ObjCBlockMember
// CHECK-LABEL: define linkonce_odr void @_ZN15ObjCBlockMemberC2Ev
// CHECK: store {{.*}} null,
// CHECK-NEXT: ret void

// Implicitly-generated destructor for ObjCBlockMember
// CHECK-LABEL: define linkonce_odr void @_ZN15ObjCBlockMemberD2Ev
// CHECK: call void @llvm.objc.storeStrong(i8*
// CHECK: ret

// Implicitly-generated copy constructor for ObjCBlockMember
// CHECK-LABEL: define linkonce_odr void @_ZN15ObjCBlockMemberC2ERKS_
// CHECK: call i8* @llvm.objc.retainBlock
// CHECK: ret

