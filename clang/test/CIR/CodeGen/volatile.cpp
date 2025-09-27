// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s < %t.cir
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM %s < %t-cir.ll
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG %s < %t.ll

int test_load(volatile int *ptr) {
  return *ptr;
}

// CIR: cir.func dso_local @_Z9test_loadPVi
// CIR:   cir.load volatile

// LLVM: define {{.*}} i32 @_Z9test_loadPVi
// LLVM:   load volatile i32, ptr %{{.*}}

// OGCG: define {{.*}} i32 @_Z9test_loadPVi
// OGCG:   load volatile i32, ptr %{{.*}}

void test_store(volatile int *ptr) {
  *ptr = 42;
}

// CIR: cir.func dso_local @_Z10test_storePVi
// CIR:   cir.store volatile

// LLVM: define {{.*}} void @_Z10test_storePVi
// LLVM:   store volatile i32 42, ptr %{{.*}}

// OGCG: define {{.*}} void @_Z10test_storePVi
// OGCG:   store volatile i32 42, ptr %{{.*}}

struct Foo {
  int x;
  volatile int y;
  volatile int z: 4;
};

int test_load_field1(volatile Foo *ptr) {
  return ptr->x;
}

// CIR: cir.func dso_local @_Z16test_load_field1PV3Foo
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member
// CIR:   %{{.+}} = cir.load volatile{{.*}} %[[MEMBER_ADDR]]

// LLVM: define {{.*}} i32 @_Z16test_load_field1PV3Foo
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.Foo, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %{{.*}} = load volatile i32, ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} i32 @_Z16test_load_field1PV3Foo
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.Foo, ptr %{{.*}}, i32 0, i32 0
// OGCG:   %{{.*}} = load volatile i32, ptr %[[MEMBER_ADDR]]

int test_load_field2(Foo *ptr) {
  return ptr->y;
}

// CIR: cir.func dso_local @_Z16test_load_field2P3Foo
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member
// CIR:   %{{.+}} = cir.load volatile{{.*}} %[[MEMBER_ADDR]]

// LLVM: define {{.*}} i32 @_Z16test_load_field2P3Foo
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.Foo, ptr %{{.*}}, i32 0, i32 1
// LLVM:   %{{.*}} = load volatile i32, ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} i32 @_Z16test_load_field2P3Foo
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.Foo, ptr %{{.*}}, i32 0, i32 1
// OGCG:   %{{.*}} = load volatile i32, ptr %[[MEMBER_ADDR]]

int test_load_field3(Foo *ptr) {
  return ptr->z;
}

// CIR: cir.func dso_local @_Z16test_load_field3P3Foo
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member
// CIR:   %{{.*}} = cir.get_bitfield align(4) (#bfi_z, %[[MEMBER_ADDR:.+]] {is_volatile} : !cir.ptr<!u8i>) -> !s32i

// LLVM: define {{.*}} i32 @_Z16test_load_field3P3Foo
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.Foo, ptr %{{.*}}, i32 0, i32 2
// LLVM:   %[[TMP1:.*]] = load volatile i8, ptr %[[MEMBER_ADDR]]
// LLVM:   %[[TMP2:.*]] = shl i8 %[[TMP1]], 4
// LLVM:   %[[TMP3:.*]] = ashr i8 %[[TMP2]], 4
// LLVM:   %{{.*}} = sext i8 %[[TMP3]] to i32

// OGCG: define {{.*}} i32 @_Z16test_load_field3P3Foo
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.Foo, ptr %{{.*}}, i32 0, i32 2
// OGCG:   %[[TMP1:.*]] = load volatile i8, ptr %[[MEMBER_ADDR]]
// OGCG:   %[[TMP2:.*]] = shl i8 %[[TMP1]], 4
// OGCG:   %[[TMP3:.*]] = ashr i8 %[[TMP2]], 4
// OGCG:   %{{.*}} = sext i8 %[[TMP3]] to i32

void test_store_field1(volatile Foo *ptr) {
  ptr->x = 42;
}

// CIR: cir.func dso_local @_Z17test_store_field1PV3Foo
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member
// CIR:   cir.store volatile{{.*}} %{{.+}}, %[[MEMBER_ADDR]]

// LLVM: define {{.*}} void @_Z17test_store_field1PV3Foo
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.Foo, ptr %{{.*}}, i32 0, i32 0
// LLVM:   store volatile i32 42, ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} void @_Z17test_store_field1PV3Foo
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.Foo, ptr %{{.*}}, i32 0, i32 0
// OGCG:   store volatile i32 42, ptr %[[MEMBER_ADDR]]

void test_store_field2(Foo *ptr) {
  ptr->y = 42;
}

// CIR: cir.func dso_local @_Z17test_store_field2P3Foo
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member
// CIR:   cir.store volatile{{.*}} %{{.+}}, %[[MEMBER_ADDR]]

// LLVM: define {{.*}} void @_Z17test_store_field2P3Foo
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.Foo, ptr %{{.*}}, i32 0, i32 1
// LLVM:   store volatile i32 42, ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} void @_Z17test_store_field2P3Foo
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.Foo, ptr %{{.*}}, i32 0, i32 1
// OGCG:   store volatile i32 42, ptr %[[MEMBER_ADDR]]

void test_store_field3(Foo *ptr) {
  ptr->z = 4;
}

// CIR: cir.func dso_local @_Z17test_store_field3P3Foo
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member
// CIR:   cir.set_bitfield align(4) (#bfi_z, %[[MEMBER_ADDR:.+]] : !cir.ptr<!u8i>, %1 : !s32i) {is_volatile}

// LLVM: define {{.*}} void @_Z17test_store_field3P3Foo
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.Foo, ptr %{{.*}}, i32 0, i32 2
// LLVM:   %[[TMP1:.*]] = load volatile i8, ptr %[[MEMBER_ADDR]]
// LLVM:   %[[TMP2:.*]] = and i8 %[[TMP1]], -16
// LLVM:   %[[TMP3:.*]] = or i8 %[[TMP2]], 4
// LLVM:   store volatile i8 %[[TMP3]], ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} void @_Z17test_store_field3P3Foo
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.Foo, ptr %{{.*}}, i32 0, i32 2
// OGCG:   %[[TMP1:.*]] = load volatile i8, ptr %[[MEMBER_ADDR]]
// OGCG:   %[[TMP2:.*]] = and i8 %[[TMP1]], -16
// OGCG:   %[[TMP3:.*]] = or i8 %[[TMP2]], 4
// OGCG:   store volatile i8 %[[TMP3]], ptr %[[MEMBER_ADDR]]

struct A {
  int x;
  void set_x(int val) volatile;
  int get_x() volatile;
};

void A::set_x(int val) volatile {
  x = val;
}

// CIR: cir.func dso_local @_ZNV1A5set_xEi
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member %{{.*}}[0] {name = "x"}
// CIR:   cir.store volatile {{.*}} %{{.*}}, %[[MEMBER_ADDR]]

// LLVM: define {{.*}} void @_ZNV1A5set_xEi
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.A, ptr %{{.*}}, i32 0, i32 0
// LLVM:   store volatile i32 %{{.*}}, ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} void @_ZNV1A5set_xEi
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.A, ptr %{{.*}}, i32 0, i32 0
// OGCG:   store volatile i32 %{{.*}}, ptr %[[MEMBER_ADDR]]

int A::get_x() volatile {
  return x;
}

// CIR: cir.func dso_local @_ZNV1A5get_xEv
// CIR:   %[[MEMBER_ADDR:.*]] = cir.get_member %{{.*}}[0] {name = "x"}
// CIR:   cir.load volatile {{.*}} %[[MEMBER_ADDR]]

// LLVM: define {{.*}} i32 @_ZNV1A5get_xEv
// LLVM:   %[[MEMBER_ADDR:.*]] = getelementptr %struct.A, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %{{.*}} = load volatile i32, ptr %[[MEMBER_ADDR]]

// OGCG: define {{.*}} i32 @_ZNV1A5get_xEv
// OGCG:   %[[MEMBER_ADDR:.*]] = getelementptr inbounds nuw %struct.A, ptr %{{.*}}, i32 0, i32 0
// OGCG:   %{{.*}} = load volatile i32, ptr %[[MEMBER_ADDR]]
