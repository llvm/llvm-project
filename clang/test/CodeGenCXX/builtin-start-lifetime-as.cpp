// RUN: %clang_cc1 -std=c++23 -O2 -emit-llvm %s -o - | FileCheck %s

struct Struct {
    int x;
    float y;
};

// {{.*}} to match for `noundef readnone returned captures(ret: address, provenance)`
// CHECK-LABEL: define {{.*}} ptr @_Z15test_start_lifePv(ptr {{.*}} %buffer)
Struct* test_start_life(void* buffer) {
    // CHECK: ret ptr %buffer
    return (Struct*)__builtin_start_lifetime_as((Struct*)buffer);
}
