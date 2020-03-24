// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -x c++ -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -x c++ -o - %s -mfloat-abi soft \
// RUN:   | FileCheck %s --check-prefix=SOFT-FLOAT

// For compatibility with GCC, this structure is passed in an FPR in C++,
// but passed in a GPR in C (checked in systemz-abi.c).

struct agg_float_cpp { float a; int : 0; };
struct agg_float_cpp pass_agg_float_cpp(struct agg_float_cpp arg) { return arg; }
// CHECK-LABEL: define void @_Z18pass_agg_float_cpp13agg_float_cpp(%struct.agg_float_cpp* noalias sret align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT:  define void @_Z18pass_agg_float_cpp13agg_float_cpp(%struct.agg_float_cpp* noalias sret align 4 %{{.*}}, i32 %{{.*}})
