// RUN: %clang_cc1 -fms-extensions -w -triple i386-pc-win32 -emit-llvm -o - %s | FileCheck %s

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
// CHECK-SAME: (ptr inalloca(<{ %struct.NonTrivial, ptr }>) %0)

int pass_inalloca_overaligned() {
  gvi32 = receive_inalloca_overaligned(NonTrivial(), OverAligned());
  return gvi32;
}

// CHECK-LABEL: define dso_local noundef i32 @"?pass_inalloca_overaligned@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.OverAligned, align 64
// CHECK: call ptr @llvm.stacksave()
// CHECK: alloca inalloca <{ %struct.NonTrivial, ptr }>

// Construct OverAligned into TMP.
// CHECK: call x86_thiscallcc noundef ptr @"??0OverAligned@@QAE@XZ"(ptr {{[^,]*}} [[TMP]])

// Construct NonTrivial into the GEP.
// CHECK: [[GEP:%[^ ]*]] = getelementptr inbounds <{ %struct.NonTrivial, ptr }>, ptr %{{.*}}, i32 0, i32 0
// CHECK: call x86_thiscallcc noundef ptr @"??0NonTrivial@@QAE@XZ"(ptr {{[^,]*}} [[GEP]])

// Store the address of an OverAligned temporary into the struct.
// CHECK: getelementptr inbounds <{ %struct.NonTrivial, ptr }>, ptr %{{.*}}, i32 0, i32 1
// CHECK: store ptr [[TMP]], ptr %{{.*}}, align 4
// CHECK: call noundef i32 @"?receive_inalloca_overaligned@@Y{{.*}}"(ptr inalloca(<{ %struct.NonTrivial, ptr }>) %argmem)

int receive_both(Both o) {
  return o.x + o.y;
}

// CHECK-LABEL: define dso_local noundef i32 @"?receive_both@@Y{{.*}}"
// CHECK-SAME: (ptr noundef %o)

int pass_both() {
  gvi32 = receive_both(Both());
  return gvi32;
}

// CHECK-LABEL: define dso_local noundef i32 @"?pass_both@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.Both, align 8
// CHECK: call x86_thiscallcc noundef ptr @"??0Both@@QAE@XZ"(ptr {{[^,]*}} [[TMP]])
// CHECK: call noundef i32 @"?receive_both@@Y{{.*}}"(ptr noundef [[TMP]])

int receive_inalloca_both(NonTrivial nt, Both o) {
  return nt.x + o.x + o.y;
}

// CHECK-LABEL: define dso_local noundef i32 @"?receive_inalloca_both@@Y{{.*}}"
// CHECK-SAME: (ptr inalloca(<{ %struct.NonTrivial, ptr }>) %0)

int pass_inalloca_both() {
  gvi32 = receive_inalloca_both(NonTrivial(), Both());
  return gvi32;
}

// CHECK-LABEL: define dso_local noundef i32 @"?pass_inalloca_both@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.Both, align 8
// CHECK: call x86_thiscallcc noundef ptr @"??0Both@@QAE@XZ"(ptr {{[^,]*}} [[TMP]])
// CHECK: call noundef i32 @"?receive_inalloca_both@@Y{{.*}}"(ptr inalloca(<{ %struct.NonTrivial, ptr }>) %argmem)

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
// CHECK-SAME: (ptr noundef %o)

int passMyPtr() { return receiveMyPtr(MyPtr()); }

// CHECK-LABEL: define dso_local noundef i32 @"?passMyPtr@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.MyPtr, align 8
// CHECK: call x86_thiscallcc noundef ptr @"??0MyPtr@@QAE@XZ"(ptr {{[^,]*}} [[TMP]])
// CHECK: call noundef i32 @"?receiveMyPtr@@Y{{.*}}"(ptr noundef [[TMP]])
