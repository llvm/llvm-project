// RUN: %clang_cc1 -no-opaque-pointers -fms-extensions -w -triple i386-pc-win32 -emit-llvm -o - %s | FileCheck %s

// PR44395
// MSVC passes overaligned types indirectly since MSVC 2015. Make sure that
// works with inalloca.

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &o);
  int x;
};

struct __declspec(align(64)) OverAligned {
  OverAligned();
  int buf[16];
};

struct __declspec(align(8)) Both {
  Both();
  Both(const Both &o);
  int x, y;
};

extern int gvi32;

int receive_inalloca_overaligned(NonTrivial nt, OverAligned o) {
  return nt.x + o.buf[0];
}

// CHECK-LABEL: define dso_local noundef i32 @"?receive_inalloca_overaligned@@Y{{.*}}"
// CHECK-SAME: (<{ %struct.NonTrivial, %struct.OverAligned* }>* inalloca(<{ %struct.NonTrivial, %struct.OverAligned* }>) %0)

int pass_inalloca_overaligned() {
  gvi32 = receive_inalloca_overaligned(NonTrivial(), OverAligned());
  return gvi32;
}

// CHECK-LABEL: define dso_local noundef i32 @"?pass_inalloca_overaligned@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.OverAligned, align 64
// CHECK: call i8* @llvm.stacksave()
// CHECK: alloca inalloca <{ %struct.NonTrivial, %struct.OverAligned* }>

// Construct OverAligned into TMP.
// CHECK: call x86_thiscallcc noundef %struct.OverAligned* @"??0OverAligned@@QAE@XZ"(%struct.OverAligned* {{[^,]*}} [[TMP]])

// Construct NonTrivial into the GEP.
// CHECK: [[GEP:%[^ ]*]] = getelementptr inbounds <{ %struct.NonTrivial, %struct.OverAligned* }>, <{ %struct.NonTrivial, %struct.OverAligned* }>* %{{.*}}, i32 0, i32 0
// CHECK: call x86_thiscallcc noundef %struct.NonTrivial* @"??0NonTrivial@@QAE@XZ"(%struct.NonTrivial* {{[^,]*}} [[GEP]])

// Store the address of an OverAligned temporary into the struct.
// CHECK: getelementptr inbounds <{ %struct.NonTrivial, %struct.OverAligned* }>, <{ %struct.NonTrivial, %struct.OverAligned* }>* %{{.*}}, i32 0, i32 1
// CHECK: store %struct.OverAligned* [[TMP]], %struct.OverAligned** %{{.*}}, align 4
// CHECK: call noundef i32 @"?receive_inalloca_overaligned@@Y{{.*}}"(<{ %struct.NonTrivial, %struct.OverAligned* }>* inalloca(<{ %struct.NonTrivial, %struct.OverAligned* }>) %argmem)

int receive_both(Both o) {
  return o.x + o.y;
}

// CHECK-LABEL: define dso_local noundef i32 @"?receive_both@@Y{{.*}}"
// CHECK-SAME: (%struct.Both* noundef %o)

int pass_both() {
  gvi32 = receive_both(Both());
  return gvi32;
}

// CHECK-LABEL: define dso_local noundef i32 @"?pass_both@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.Both, align 8
// CHECK: call x86_thiscallcc noundef %struct.Both* @"??0Both@@QAE@XZ"(%struct.Both* {{[^,]*}} [[TMP]])
// CHECK: call noundef i32 @"?receive_both@@Y{{.*}}"(%struct.Both* noundef [[TMP]])

int receive_inalloca_both(NonTrivial nt, Both o) {
  return nt.x + o.x + o.y;
}

// CHECK-LABEL: define dso_local noundef i32 @"?receive_inalloca_both@@Y{{.*}}"
// CHECK-SAME: (<{ %struct.NonTrivial, %struct.Both* }>* inalloca(<{ %struct.NonTrivial, %struct.Both* }>) %0)

int pass_inalloca_both() {
  gvi32 = receive_inalloca_both(NonTrivial(), Both());
  return gvi32;
}

// CHECK-LABEL: define dso_local noundef i32 @"?pass_inalloca_both@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.Both, align 8
// CHECK: call x86_thiscallcc noundef %struct.Both* @"??0Both@@QAE@XZ"(%struct.Both* {{[^,]*}} [[TMP]])
// CHECK: call noundef i32 @"?receive_inalloca_both@@Y{{.*}}"(<{ %struct.NonTrivial, %struct.Both* }>* inalloca(<{ %struct.NonTrivial, %struct.Both* }>) %argmem)

// Here we have a type that is:
// - overaligned
// - not trivially copyable
// - can be "passed in registers" due to [[trivial_abi]]
// Clang should pass it directly.
struct [[trivial_abi]] alignas(8) MyPtr {
  MyPtr();
  MyPtr(const MyPtr &o);
  ~MyPtr();
  int *ptr;
};

int receiveMyPtr(MyPtr o) { return *o.ptr; }

// CHECK-LABEL: define dso_local noundef i32 @"?receiveMyPtr@@Y{{.*}}"
// CHECK-SAME: (%struct.MyPtr* noundef %o)

int passMyPtr() { return receiveMyPtr(MyPtr()); }

// CHECK-LABEL: define dso_local noundef i32 @"?passMyPtr@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.MyPtr, align 8
// CHECK: call x86_thiscallcc noundef %struct.MyPtr* @"??0MyPtr@@QAE@XZ"(%struct.MyPtr* {{[^,]*}} [[TMP]])
// CHECK: call noundef i32 @"?receiveMyPtr@@Y{{.*}}"(%struct.MyPtr* noundef [[TMP]])
