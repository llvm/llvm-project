// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm -x c++ -o - %s | FileCheck %s

struct empty { };
struct agg_nofloat_empty { float a; empty dummy; };
struct complex_like_agg_nofloat_empty { struct agg_nofloat_empty a; struct agg_nofloat_empty b; };
struct complex_like_agg_nofloat_empty pass_complex_like_agg_nofloat_empty(struct complex_like_agg_nofloat_empty arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @_Z35pass_complex_like_agg_nofloat_empty30complex_like_agg_nofloat_empty([2 x i64] %{{.*}})

struct agg_float_empty { float a; [[no_unique_address]] empty dummy; };
struct complex_like_agg_float_empty { struct agg_float_empty a; struct agg_float_empty b; };
struct complex_like_agg_float_empty pass_complex_like_agg_float_empty(struct complex_like_agg_float_empty arg) { return arg; }
// CHECK-LABEL: define { float, float } @_Z33pass_complex_like_agg_float_empty28complex_like_agg_float_empty({ float, float } %{{.*}})

struct noemptybase { empty dummy; };
struct agg_nofloat_emptybase : noemptybase { float a; };
struct complex_like_agg_nofloat_emptybase { struct agg_nofloat_emptybase a; struct agg_nofloat_emptybase b; };
struct complex_like_agg_nofloat_emptybase pass_agg_nofloat_emptybase(struct complex_like_agg_nofloat_emptybase arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @_Z26pass_agg_nofloat_emptybase34complex_like_agg_nofloat_emptybase([2 x i64] %{{.*}})

struct emptybase { [[no_unique_address]] empty dummy; };
struct agg_float_emptybase : emptybase { float a; };
struct complex_like_agg_float_emptybase { struct agg_float_emptybase a; struct agg_float_emptybase b; };
struct complex_like_agg_float_emptybase pass_agg_float_emptybase(struct complex_like_agg_float_emptybase arg) { return arg; }
// CHECK-LABEL: define { float, float } @_Z24pass_agg_float_emptybase32complex_like_agg_float_emptybase({ float, float } %{{.*}})
