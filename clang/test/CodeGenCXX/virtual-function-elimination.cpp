// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -flto -flto-unit -fvirtual-function-elimination -fwhole-program-vtables -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -Xclang -no-opaque-pointers -target x86_64-unknown-linux -flto -fvirtual-function-elimination -fno-virtual-function-elimination -fwhole-program-vtables -S -emit-llvm -o - %s | FileCheck %s -check-prefix=NOVFE

struct __attribute__((visibility("default"))) A {
  virtual void foo();
};

void test_1(A *p) {
  // A has default visibility, so no need for type.checked.load.
// CHECK-LABEL: define{{.*}} void @_Z6test_1P1A
// NOVFE-LABEL: define dso_local void @_Z6test_1P1A
// CHECK: [[FN_PTR_ADDR:%.+]] = getelementptr inbounds void (%struct.A*)*, void (%struct.A*)** {{%.+}}, i64 0
// NOVFE: [[FN_PTR_ADDR:%.+]] = getelementptr inbounds void (%struct.A*)*, void (%struct.A*)** {{%.+}}, i64 0
// CHECK: [[FN_PTR:%.+]] = load void (%struct.A*)*, void (%struct.A*)** [[FN_PTR_ADDR]]
// NOVFE: [[FN_PTR:%.+]] = load void (%struct.A*)*, void (%struct.A*)** [[FN_PTR_ADDR]]
// CHECK: call void [[FN_PTR]](
// NOVFE: call void [[FN_PTR]](
  p->foo();
}


struct __attribute__((visibility("hidden"))) [[clang::lto_visibility_public]] B {
  virtual void foo();
};

void test_2(B *p) {
  // B has public LTO visibility, so no need for type.checked.load.
// CHECK-LABEL: define{{.*}} void @_Z6test_2P1B
// NOVFE-LABEL: define dso_local void @_Z6test_2P1B
// CHECK: [[FN_PTR_ADDR:%.+]] = getelementptr inbounds void (%struct.B*)*, void (%struct.B*)** {{%.+}}, i64 0
// NOVFE: [[FN_PTR_ADDR:%.+]] = getelementptr inbounds void (%struct.B*)*, void (%struct.B*)** {{%.+}}, i64 0
// CHECK: [[FN_PTR:%.+]] = load void (%struct.B*)*, void (%struct.B*)** [[FN_PTR_ADDR]]
// NOVFE: [[FN_PTR:%.+]] = load void (%struct.B*)*, void (%struct.B*)** [[FN_PTR_ADDR]]
// CHECK: call void [[FN_PTR]](
// NOVFE: call void [[FN_PTR]](
  p->foo();
}


struct __attribute__((visibility("hidden"))) C {
  virtual void foo();
  virtual void bar();
};

void test_3(C *p) {
  // C has hidden visibility, so we generate type.checked.load to allow VFE.
// CHECK-LABEL: define{{.*}} void @_Z6test_3P1C
// NOVFE-LABEL: define dso_local void @_Z6test_3P1C
// CHECK: [[LOAD:%.+]] = call { i8*, i1 } @llvm.type.checked.load(i8* {{%.+}}, i32 0, metadata !"_ZTS1C")
// NOVFE: call i1 @llvm.type.test(i8* {{%.+}}, metadata !"_ZTS1C")
// CHECK: [[FN_PTR_I8:%.+]] = extractvalue { i8*, i1 } [[LOAD]], 0
// NOVFE: [[FN_PTR:%.+]] = load void (%struct.C*)*, void (%struct.C*)** {{%.+}}, align 8
// CHECK: [[FN_PTR:%.+]] = bitcast i8* [[FN_PTR_I8]] to void (%struct.C*)*
// CHECK: call void [[FN_PTR]](
// NOVFE: call void [[FN_PTR]](
  p->foo();
}

void test_4(C *p) {
  // When using type.checked.load, we pass the vtable offset to the intrinsic,
  // rather than adding it to the pointer with a GEP.
// CHECK-LABEL: define{{.*}} void @_Z6test_4P1C
// NOVFE-LABEL: define dso_local void @_Z6test_4P1C
// CHECK: [[LOAD:%.+]] = call { i8*, i1 } @llvm.type.checked.load(i8* {{%.+}}, i32 8, metadata !"_ZTS1C")
// NOVFE: call i1 @llvm.type.test(i8* {{%.+}}, metadata !"_ZTS1C")
// CHECK: [[FN_PTR_I8:%.+]] = extractvalue { i8*, i1 } [[LOAD]], 0
// NOVFE: [[FN_PTR:%.+]] = load void (%struct.C*)*, void (%struct.C*)** {{%.+}}, align 8
// CHECK: [[FN_PTR:%.+]] = bitcast i8* [[FN_PTR_I8]] to void (%struct.C*)*
// CHECK: call void [[FN_PTR]](
// NOVFE: call void [[FN_PTR]](
  p->bar();
}

void test_5(C *p, void (C::*q)(void)) {
  // We also use type.checked.load for the virtual side of member function
  // pointer calls. We use a GEP to calculate the address to load from and pass
  // 0 as the offset to the intrinsic, because we know that the load must be
  // from exactly the point marked by one of the function-type metadatas (in
  // this case "_ZTSM1CFvvE.virtual"). If we passed the offset from the member
  // function pointer to the intrinsic, this information would be lost. No
  // codegen changes on the non-virtual side.
// CHECK-LABEL: define{{.*}} void @_Z6test_5P1CMS_FvvE(
// NOVFE-LABEL: define dso_local void @_Z6test_5P1CMS_FvvE(
// CHECK: [[FN_PTR_ADDR:%.+]] = getelementptr i8, i8* %vtable, i64 {{%.+}}
// CHECK: [[LOAD:%.+]] = call { i8*, i1 } @llvm.type.checked.load(i8* [[FN_PTR_ADDR]], i32 0, metadata !"_ZTSM1CFvvE.virtual")
// NOVFE-NOT: call { i8*, i1 } @llvm.type.checked.load(i8* {{%.+}}, i32 0, metadata !"_ZTSM1CFvvE.virtual")
// CHECK: [[FN_PTR_I8:%.+]] = extractvalue { i8*, i1 } [[LOAD]], 0
// CHECK: [[FN_PTR:%.+]] = bitcast i8* [[FN_PTR_I8]] to void (%struct.C*)*
// NOVFE: [[FN_PTR:%.+]] = load void (%struct.C*)*, void (%struct.C*)** {{%.+}}, align 8

// CHECK: [[PHI:%.+]] = phi void (%struct.C*)* {{.*}}[ [[FN_PTR]], {{.*}} ]
// NOVFE: [[PHI:%.+]] = phi void (%struct.C*)* {{.*}}[ [[FN_PTR]], {{.*}} ]
// CHECK: call void [[PHI]](
// NOVFE: call void [[PHI]](
  (p->*q)();
}
