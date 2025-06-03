// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -x c++ -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -x c++ -o - %s -mfloat-abi soft \
// RUN:   | FileCheck %s --check-prefix=SOFT-FLOAT

// Verify that class types are also recognized as float-like aggregate types

class agg_float_class { float a; };
class agg_float_class pass_agg_float_class(class agg_float_class arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z20pass_agg_float_class15agg_float_class(ptr dead_on_unwind noalias writable sret(%class.agg_float_class) align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT-LABEL: define{{.*}} void @_Z20pass_agg_float_class15agg_float_class(ptr dead_on_unwind noalias writable sret(%class.agg_float_class) align 4 %{{.*}}, i32 noext %{{.*}})

class agg_double_class { double a; };
class agg_double_class pass_agg_double_class(class agg_double_class arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z21pass_agg_double_class16agg_double_class(ptr dead_on_unwind noalias writable sret(%class.agg_double_class) align 8 %{{.*}}, double %{{.*}})
// SOFT-FLOAT-LABEL: define{{.*}} void @_Z21pass_agg_double_class16agg_double_class(ptr dead_on_unwind noalias writable sret(%class.agg_double_class) align 8 %{{.*}}, i64 %{{.*}})


// This structure is passed in a GPR in C++ (and C, checked in systemz-abi.c).
struct agg_float_cpp { float a; int : 0; };
struct agg_float_cpp pass_agg_float_cpp(struct agg_float_cpp arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z18pass_agg_float_cpp13agg_float_cpp(ptr dead_on_unwind noalias writable sret(%struct.agg_float_cpp) align 4 %{{.*}}, i32 noext %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z18pass_agg_float_cpp13agg_float_cpp(ptr dead_on_unwind noalias writable sret(%struct.agg_float_cpp) align 4 %{{.*}}, i32 noext %{{.*}})


// A field member of empty class type in C++ makes the record nonhomogeneous,
// unless it is marked as [[no_unique_address]].  This does not apply to arrays.
struct empty { };
struct agg_nofloat_empty { float a; empty dummy; };
struct agg_nofloat_empty pass_agg_nofloat_empty(struct agg_nofloat_empty arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z22pass_agg_nofloat_empty17agg_nofloat_empty(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_empty) align 4 %{{.*}}, i64 %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z22pass_agg_nofloat_empty17agg_nofloat_empty(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_empty) align 4 %{{.*}}, i64 %{{.*}})
struct agg_float_empty { float a; [[no_unique_address]] empty dummy; };
struct agg_float_empty pass_agg_float_empty(struct agg_float_empty arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z20pass_agg_float_empty15agg_float_empty(ptr dead_on_unwind noalias writable sret(%struct.agg_float_empty) align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z20pass_agg_float_empty15agg_float_empty(ptr dead_on_unwind noalias writable sret(%struct.agg_float_empty) align 4 %{{.*}}, i32 noext %{{.*}})
struct agg_nofloat_emptyarray { float a; [[no_unique_address]] empty dummy[3]; };
struct agg_nofloat_emptyarray pass_agg_nofloat_emptyarray(struct agg_nofloat_emptyarray arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z27pass_agg_nofloat_emptyarray22agg_nofloat_emptyarray(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_emptyarray) align 4 %{{.*}}, i64 %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z27pass_agg_nofloat_emptyarray22agg_nofloat_emptyarray(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_emptyarray) align 4 %{{.*}}, i64 %{{.*}})

// And likewise for members of base classes.
struct noemptybase { empty dummy; };
struct agg_nofloat_emptybase : noemptybase { float a; };
struct agg_nofloat_emptybase pass_agg_nofloat_emptybase(struct agg_nofloat_emptybase arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_emptybase) align 4 %{{.*}}, i64 %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_emptybase) align 4 %{{.*}}, i64 %{{.*}})
struct emptybase { [[no_unique_address]] empty dummy; };
struct agg_float_emptybase : emptybase { float a; };
struct agg_float_emptybase pass_agg_float_emptybase(struct agg_float_emptybase arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z24pass_agg_float_emptybase19agg_float_emptybase(ptr dead_on_unwind noalias writable sret(%struct.agg_float_emptybase) align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z24pass_agg_float_emptybase19agg_float_emptybase(ptr dead_on_unwind noalias writable sret(%struct.agg_float_emptybase) align 4 %{{.*}}, i32 noext %{{.*}})
struct noemptybasearray { [[no_unique_address]] empty dummy[3]; };
struct agg_nofloat_emptybasearray : noemptybasearray { float a; };
struct agg_nofloat_emptybasearray pass_agg_nofloat_emptybasearray(struct agg_nofloat_emptybasearray arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @_Z31pass_agg_nofloat_emptybasearray26agg_nofloat_emptybasearray(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_emptybasearray) align 4 %{{.*}}, i64 %{{.*}})
// SOFT-FLOAT-LABEL:  define{{.*}} void @_Z31pass_agg_nofloat_emptybasearray26agg_nofloat_emptybasearray(ptr dead_on_unwind noalias writable sret(%struct.agg_nofloat_emptybasearray) align 4 %{{.*}}, i64 %{{.*}})

