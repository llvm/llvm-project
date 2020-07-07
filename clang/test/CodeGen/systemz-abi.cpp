// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -x c++ -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -x c++ -o - %s -mfloat-abi soft \
// RUN:   | FileCheck %s --check-prefix=SOFT-FLOAT

// Verify that class types are also recognized as float-like aggregate types

class agg_float_class { float a; };
class agg_float_class pass_agg_float_class(class agg_float_class arg) { return arg; }
// CHECK-LABEL: define void @_Z20pass_agg_float_class15agg_float_class(%class.agg_float_class* noalias sret align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT-LABEL: define void @_Z20pass_agg_float_class15agg_float_class(%class.agg_float_class* noalias sret align 4 %{{.*}}, i32 %{{.*}})

class agg_double_class { double a; };
class agg_double_class pass_agg_double_class(class agg_double_class arg) { return arg; }
// CHECK-LABEL: define void @_Z21pass_agg_double_class16agg_double_class(%class.agg_double_class* noalias sret align 8 %{{.*}}, double %{{.*}})
// SOFT-FLOAT-LABEL: define void @_Z21pass_agg_double_class16agg_double_class(%class.agg_double_class* noalias sret align 8 %{{.*}}, i64 %{{.*}})


// For compatibility with GCC, this structure is passed in an FPR in C++,
// but passed in a GPR in C (checked in systemz-abi.c).

struct agg_float_cpp { float a; int : 0; };
struct agg_float_cpp pass_agg_float_cpp(struct agg_float_cpp arg) { return arg; }
// CHECK-LABEL: define void @_Z18pass_agg_float_cpp13agg_float_cpp(%struct.agg_float_cpp* noalias sret align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT-LABEL:  define void @_Z18pass_agg_float_cpp13agg_float_cpp(%struct.agg_float_cpp* noalias sret align 4 %{{.*}}, i32 %{{.*}})

