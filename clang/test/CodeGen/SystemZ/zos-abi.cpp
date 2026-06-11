// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm -no-enable-noundef-analysis -x c++ -o - %s | FileCheck %s

// Verify that class types are recognized as float-like aggregate types and passed in GPR.

class agg_float_class { float a; };
class agg_float_class pass_agg_float_class(class agg_float_class arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z20pass_agg_float_class15agg_float_class(i64 %arg.coerce)

class agg_double_class { double a; };
class agg_double_class pass_agg_double_class(class agg_double_class arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z21pass_agg_double_class16agg_double_class(i64 %arg.coerce)


// This structure is passed in also in GPR.
struct agg_float_cpp { float a; int : 0; };
struct agg_float_cpp pass_agg_float_cpp(struct agg_float_cpp arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z18pass_agg_float_cpp13agg_float_cpp(i64 %arg.coerce)

// In C++ a  data member of empty class type makes the record nonhomogeneous,
// regardless if it's marked as [[no_unique_address]] or not.
struct empty { };
struct agg_nofloat_empty { float a; empty dummy; };
struct agg_nofloat_empty pass_agg_nofloat_empty(struct agg_nofloat_empty arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z22pass_agg_nofloat_empty17agg_nofloat_empty(i64 %arg.coerce)
struct complex_like_agg_nofloat_empty { struct agg_nofloat_empty a; struct agg_nofloat_empty b; };
struct complex_like_agg_nofloat_empty pass_complex_like_agg_nofloat_empty(struct complex_like_agg_nofloat_empty arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @_Z35pass_complex_like_agg_nofloat_empty30complex_like_agg_nofloat_empty([2 x i64] %{{.*}})

struct agg_float_empty { float a; [[no_unique_address]] empty dummy; };
struct agg_float_empty pass_agg_float_empty(struct agg_float_empty arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z20pass_agg_float_empty15agg_float_empty(i64 %arg.coerce)
struct complex_like_agg_float_empty { struct agg_float_empty a; struct agg_float_empty b; };
struct complex_like_agg_float_empty pass_complex_like_agg_float_empty(struct complex_like_agg_float_empty arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z33pass_complex_like_agg_float_empty28complex_like_agg_float_empty(i64 %{{.*}})

struct agg_nofloat_emptyarray { float a; [[no_unique_address]] empty dummy[3]; };
struct agg_nofloat_emptyarray pass_agg_nofloat_emptyarray(struct agg_nofloat_emptyarray arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z27pass_agg_nofloat_emptyarray22agg_nofloat_emptyarray(i64 %arg.coerce)


// And likewise for members of base classes.
struct noemptybase { empty dummy; };
struct agg_nofloat_emptybase : noemptybase { float a; };
struct agg_nofloat_emptybase pass_agg_nofloat_emptybase(struct agg_nofloat_emptybase arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z26pass_agg_nofloat_emptybase21agg_nofloat_emptybase(i64 %arg.coerce)
struct complex_like_agg_nofloat_emptybase { struct agg_nofloat_emptybase a; struct agg_nofloat_emptybase b; };
struct complex_like_agg_nofloat_emptybase pass_agg_nofloat_emptybase(struct complex_like_agg_nofloat_emptybase arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @_Z26pass_agg_nofloat_emptybase34complex_like_agg_nofloat_emptybase([2 x i64] %{{.*}})

struct emptybase { [[no_unique_address]] empty dummy; };
struct agg_float_emptybase : emptybase { float a; };
struct agg_float_emptybase pass_agg_float_emptybase(struct agg_float_emptybase arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z24pass_agg_float_emptybase19agg_float_emptybase(i64 %arg.coerce)
struct complex_like_agg_float_emptybase { struct agg_float_emptybase a; struct agg_float_emptybase b; };
struct complex_like_agg_float_emptybase pass_agg_float_emptybase(struct complex_like_agg_float_emptybase arg) { return arg; }
// CHECK-LABEL: define %struct.complex_like_agg_float_emptybase @_Z24pass_agg_float_emptybase32complex_like_agg_float_emptybase({ float, float } %{{.*}})

struct noemptybasearray { [[no_unique_address]] empty dummy[3]; };
struct agg_nofloat_emptybasearray : noemptybasearray { float a; };
struct agg_nofloat_emptybasearray pass_agg_nofloat_emptybasearray(struct agg_nofloat_emptybasearray arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @_Z31pass_agg_nofloat_emptybasearray26agg_nofloat_emptybasearray(i64 %{{.*}})

using D = double;
using E = __attribute__((aligned(32))) D; // attribute inside the alias
struct complexlike_alias {
  E x; // Using alias with attributed underlying type
  double y;
};
struct complexlike_alias pass_complexlike_alias(struct complexlike_alias arg) { return arg; }
// CHECK-LABEL: define %struct.complexlike_alias @_Z22pass_complexlike_alias17complexlike_alias({ double, double } %{{.*}})

// ============================================================================
// Complex-like struct using alignas specifier
// ============================================================================

struct S_A {
  double re alignas(32);;
  double im;
};

struct S_A pass_S_A(struct S_A arg) { return arg; }
// CHECK-LABEL: define void @_Z8pass_S_A3S_A(
// CHECK-SAME: ptr {{.*}} sret(%struct.S_A) align 32
// CHECK-SAME: [4 x i64] 
// CHECK: ret void

struct alignas(32) S_B {
  double re;
  double im;
};

struct S_B pass_S_B(struct S_B arg) { return arg; }
// CHECK-LABEL: define void @_Z8pass_S_B3S_B(
// CHECK-SAME: ptr {{.*}} sret(%struct.S_B) align 32
// CHECK-SAME: [4 x i64] 
// CHECK: ret void
