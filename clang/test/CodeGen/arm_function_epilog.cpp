// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-none-linux-androideabi -target-abi aapcs-linux -mfloat-abi hard -x c++ -emit-llvm %s -o - | FileCheck %s

struct Vec2 {
    union { struct { float x, y; };
            float data[2];
    };
};

// CHECK: define arm_aapcs_vfpcc %struct.Vec2 @_Z7getVec2v()
// CHECK: ret %struct.Vec2
Vec2 getVec2() {
    Vec2 out;
    union { Vec2* v; unsigned char* u; } x;
    x.v = &out;
    return out;
}
