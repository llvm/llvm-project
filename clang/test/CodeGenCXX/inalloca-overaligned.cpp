// RUN: %clang_cc1 -fms-extensions -w -triple i386-pc-win32 -emit-llvm -o - %s | FileCheck %s

// PR44395
// MSVC passes overaligned types indirectly since MSVC 2015. Make sure that
// works with inalloca.

// FIXME: Pass non-trivial *and* overaligned types indirectly. Right now the C++
// ABI rules say to use inalloca, and they take precedence, so it's not easy to
// implement this.


struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &o);
  int x;
};

struct __declspec(align(64)) OverAligned {
  OverAligned();
  int buf[16];
};

extern int gvi32;

int receive_inalloca_overaligned(NonTrivial nt, OverAligned o) {
  return nt.x + o.buf[0];
}

// CHECK-LABEL: define dso_local i32 @"?receive_inalloca_overaligned@@Y{{.*}}"
// CHECK-SAME: (<{ %struct.NonTrivial, %struct.OverAligned* }>* inalloca %0)

int pass_inalloca_overaligned() {
  gvi32 = receive_inalloca_overaligned(NonTrivial(), OverAligned());
  return gvi32;
}

// CHECK-LABEL: define dso_local i32 @"?pass_inalloca_overaligned@@Y{{.*}}"
// CHECK: [[TMP:%[^ ]*]] = alloca %struct.OverAligned, align 64
// CHECK: call i8* @llvm.stacksave()
// CHECK: alloca inalloca <{ %struct.NonTrivial, %struct.OverAligned* }>

// Construct OverAligned into TMP.
// CHECK: call x86_thiscallcc %struct.OverAligned* @"??0OverAligned@@QAE@XZ"(%struct.OverAligned* [[TMP]])

// Construct NonTrivial into the GEP.
// CHECK: [[GEP:%[^ ]*]] = getelementptr inbounds <{ %struct.NonTrivial, %struct.OverAligned* }>, <{ %struct.NonTrivial, %struct.OverAligned* }>* %{{.*}}, i32 0, i32 0
// CHECK: call x86_thiscallcc %struct.NonTrivial* @"??0NonTrivial@@QAE@XZ"(%struct.NonTrivial* [[GEP]])

// Store the address of an OverAligned temporary into the struct.
// CHECK: getelementptr inbounds <{ %struct.NonTrivial, %struct.OverAligned* }>, <{ %struct.NonTrivial, %struct.OverAligned* }>* %{{.*}}, i32 0, i32 1
// CHECK: store %struct.OverAligned* [[TMP]], %struct.OverAligned** %{{.*}}, align 4
// CHECK: call i32 @"?receive_inalloca_overaligned@@Y{{.*}}"(<{ %struct.NonTrivial, %struct.OverAligned* }>* inalloca %argmem)
