// RUN: %clang -target dxil-pc-shadermodel6.0-compute -S -o - %s | FileCheck %s
// RUN: %clang -target dxil-pc-shadermodel6.3-library -S -o - %s | FileCheck %s

// Verify that internal linkage unused functions are removed

RWBuffer<unsigned> buf;

// Never called functions should be removed.
// CHECK-NOT: define{{.*}}uncalledFor
void uncalledFor() {
     buf[1] = 1;
}

// Never called but exported functions should remain.
// CHECK: define void @"?exported@@YAXXZ"()
export void exported() {
     buf[1] = 1;
}

// Never called but noinlined functions should remain.
// CHECK: define internal void @"?noinlined@@YAXXZ"()
__attribute__((noinline)) void noinlined() {
     buf[1] = 1;
}

// Called functions marked noinline should remain.
// CHECK: define internal void @"?calledAndNoinlined@@YAXXZ"()
__attribute__((noinline)) void calledAndNoinlined() {
     buf[1] = 1;
}

// Called functions that get inlined by default should be removed.
// CHECK-NOT: define{{.*}}calledAndInlined
void calledAndInlined() {
     buf[1] = 1;
}


// Entry point functions should remain.
// CHECK: define{{.*}}main
[numthreads(1,1,1)]
[shader("compute")]
void main() {
     calledAndInlined();
     calledAndNoinlined();
     buf[0] = 0;
}