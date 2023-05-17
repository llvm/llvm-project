// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -x c++ \
// RUN:   -o - %s | FileCheck %s -check-prefix=CHECK-BE
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -x c++ \
// RUN:   -o - %s | FileCheck %s -check-prefix=CHECK-LE

class agg_float_class { float a; };
// CHECK-BE-LABEL: define{{.*}} void @_Z20pass_agg_float_class15agg_float_class(ptr noalias sret(%class.agg_float_class) align 4 %{{.*}}, float inreg %{{.*}})
// CHECK-LE-LABEL: define{{.*}} [1 x float] @_Z20pass_agg_float_class15agg_float_class(float inreg %{{.*}})
agg_float_class pass_agg_float_class(agg_float_class arg) { return arg; }

class agg_double_class { double a; };
// CHECK-BE-LABEL: define{{.*}} void @_Z21pass_agg_double_class16agg_double_class(ptr noalias sret(%class.agg_double_class) align 8 %{{.*}}, double inreg %{{.*}})
// CHECK-LE-LABEL: define{{.*}} [1 x double] @_Z21pass_agg_double_class16agg_double_class(double inreg %{{.*}})
agg_double_class pass_agg_double_class(agg_double_class arg) { return arg; }

struct agg_float_cpp { float a; int : 0; };
// CHECK-BE-LABEL: define{{.*}} void @_Z18pass_agg_float_cpp13agg_float_cpp(ptr noalias sret(%struct.agg_float_cpp) align 4 %{{.*}}, float inreg %{{.*}})
// CHECK-LE-LABEL: define{{.*}} [1 x float] @_Z18pass_agg_float_cpp13agg_float_cpp(float inreg %{{.*}})
agg_float_cpp pass_agg_float_cpp(agg_float_cpp arg) { return arg; }

struct empty { };
struct agg_nofloat_empty { float a; empty dummy; };
// CHECK-BE-LABEL: define{{.*}} void @_Z22pass_agg_nofloat_empty17agg_nofloat_empty(ptr noalias sret(%struct.agg_nofloat_empty) align 4 %{{.*}}, i64 %{{.*}})
// CHECK-LE-LABEL: define{{.*}} i64 @_Z22pass_agg_nofloat_empty17agg_nofloat_empty(i64 %{{.*}})
agg_nofloat_empty pass_agg_nofloat_empty(agg_nofloat_empty arg) { return arg; }

struct agg_float_empty { float a; [[no_unique_address]] empty dummy; };
// CHECK-BE-LABEL: define{{.*}} void @_Z20pass_agg_float_empty15agg_float_empty(ptr noalias sret(%struct.agg_float_empty) align 4 %{{.*}}, float inreg %{{.*}})
// CHECK-LE-LABEL: define{{.*}} [1 x float] @_Z20pass_agg_float_empty15agg_float_empty(float inreg %{{.*}})
agg_float_empty pass_agg_float_empty(agg_float_empty arg) { return arg; }

struct agg_nofloat_emptyarray { float a; [[no_unique_address]] empty dummy[3]; };
// CHECK-BE-LABEL: define{{.*}} void @_Z27pass_agg_nofloat_emptyarray22agg_nofloat_emptyarray(ptr noalias sret(%struct.agg_nofloat_emptyarray) align 4 %{{.*}}, i64 %{{.*}})
// CHECK-LE-LABEL: define{{.*}} i64 @_Z27pass_agg_nofloat_emptyarray22agg_nofloat_emptyarray(i64 %{{.*}})
agg_nofloat_emptyarray pass_agg_nofloat_emptyarray(agg_nofloat_emptyarray arg) { return arg; }

struct noemptybase { empty dummy; };
struct agg_nofloat_emptybase : noemptybase { float a; };
// CHECK-BE-LABEL: define{{.*}} void @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(ptr noalias sret(%struct.agg_nofloat_emptybase) align 4 %{{.*}}, i64 %{{.*}})
// CHECK-LE-LABEL: define{{.*}} i64 @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(i64 %{{.*}})
agg_nofloat_emptybase pass_agg_nofloat_emptybase(agg_nofloat_emptybase arg) { return arg; }

struct emptybase { [[no_unique_address]] empty dummy; };
struct agg_float_emptybase : emptybase { float a; };
// CHECK-BE-LABEL: define{{.*}} void @_Z24pass_agg_float_emptybase19agg_float_emptybase(ptr noalias sret(%struct.agg_float_emptybase) align 4 %{{.*}}, float inreg %{{.*}})
// CHECK-LE-LABEL: define{{.*}} [1 x float] @_Z24pass_agg_float_emptybase19agg_float_emptybase(float inreg %{{.*}})
agg_float_emptybase pass_agg_float_emptybase(agg_float_emptybase arg) { return arg; }

struct noemptybasearray { [[no_unique_address]] empty dummy[3]; };
struct agg_nofloat_emptybasearray : noemptybasearray { float a; };
// CHECK-BE-LABEL: define{{.*}} void @_Z31pass_agg_nofloat_emptybasearray26agg_nofloat_emptybasearray(ptr noalias sret(%struct.agg_nofloat_emptybasearray) align 4 %{{.*}}, i64 %{{.*}})
// CHECK-LE-LABEL: define{{.*}} i64 @_Z31pass_agg_nofloat_emptybasearray26agg_nofloat_emptybasearray(i64 %{{.*}})
agg_nofloat_emptybasearray pass_agg_nofloat_emptybasearray(agg_nofloat_emptybasearray arg) { return arg; }

// CHECK-BE: call void @_Z24pass_agg_float_emptybase19agg_float_emptybase(ptr sret(%struct.agg_float_emptybase) align 4 %{{.*}}, float inreg %{{.*}})
// CHECK-LE: call [1 x float] @_Z24pass_agg_float_emptybase19agg_float_emptybase(float inreg %{{.*}})
void pass_agg_float_emptybase_ptr(agg_float_emptybase* arg) { pass_agg_float_emptybase(*arg); }
// CHECK-BE: call void @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(ptr sret(%struct.agg_nofloat_emptybase) align 4 %{{.*}}, i64 %{{.*}})
// CHECK-LE: call i64 @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(i64 %{{.*}})
void pass_agg_nofloat_emptybase_ptr(agg_nofloat_emptybase* arg) { pass_agg_nofloat_emptybase(*arg); }
