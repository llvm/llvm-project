// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

//CHECK-LABEL: define hidden noundef i1 @_Z14test_or_scalarbb(
//CHECK-SAME: i1 noundef [[X:%.*]], i1 noundef [[Y:%.*]]) #[[ATTR0:[0-9]+]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_OR:%.*]] = or i1 [[A:%.*]], [[B:%.*]]
//CHECK:         ret i1 [[HLSL_OR]]
bool test_or_scalar(bool x, bool y)
{
    return or(x, y);
}

//CHECK-LABEL: define hidden noundef <2 x i1> @_Z13test_or_bool2Dv2_bS_(
//CHECK-SAME: <2 x i1> noundef [[X:%.*]], <2 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_OR:%.*]] = or <2 x i1> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <2 x i1> [[HLSL_OR]]
bool2 test_or_bool2(bool2 x, bool2 y)
{
    return or(x, y);
}

//CHECK-LABEL: define hidden noundef <3 x i1> @_Z13test_or_bool3Dv3_bS_(
//CHECK-SAME: <3 x i1> noundef [[X:%.*]], <3 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_OR:%.*]] = or <3 x i1> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <3 x i1> [[HLSL_OR]]
bool3 test_or_bool3(bool3 x, bool3 y)
{
    return or(x, y);
}

//CHECK-LABEL: define hidden noundef <4 x i1> @_Z13test_or_bool4Dv4_bS_(
//CHECK-SAME: <4 x i1> noundef [[X:%.*]], <4 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_OR:%.*]] = or <4 x i1> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <4 x i1> [[HLSL_OR]]
bool4 test_or_bool4(bool4 x, bool4 y)
{
    return or(x, y);
}

//CHECK-LABEL: define hidden noundef i1 @_Z11test_or_intii(
//CHECK-SAME: i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[TOBBOL:%.*]] = icmp ne i32 [[A:%.*]], 0
//CHECK:         [[TOBBOL1:%.*]] = icmp ne i32 [[B:%.*]], 0
//CHECK:         [[HLSL_OR:%.*]] = or i1 [[TOBBOL]], [[TOBBOL1]]
//CHECK:         ret i1 [[HLSL_OR]]
bool test_or_int(int x, int y)
{
    return or(x, y);
}

//CHECK-LABEL: define hidden noundef <4 x i1> @_Z12test_or_int4Dv4_iS_(
//CHECK-SAME: <4 x i32> noundef [[X:%.*]], <4 x i32> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[TOBOOL:%.*]] = icmp ne <4 x i32> [[A:%.*]], zeroinitializer
//CHECK:         [[TOBOOL1:%.*]] = icmp ne <4 x i32> [[B:%.*]], zeroinitializer
//CHECK:         [[HLSL_OR:%.*]] = or <4 x i1> [[TOBOOL]], [[TOBOOL1]]
//CHECK:         ret <4 x i1> [[HLSL_OR]]
bool4 test_or_int4(int4 x, int4 y)
{
    return or(x, y);
}

//CHECK-LABEL: define hidden noundef <4 x i1> @_Z14test_or_float4Dv4_fS_(
//CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[X:%.*]], <4 x float> noundef nofpclass(nan inf) [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[TOBOOL:%.*]] =  fcmp reassoc nnan ninf nsz arcp afn une <4 x float> [[A:%.*]], zeroinitializer
//CHECK:         [[TOBOOL1:%.*]] = fcmp reassoc nnan ninf nsz arcp afn une <4 x float> [[B:%.*]], zeroinitializer
//CHECK:         [[HLSL_OR:%.*]] = or <4 x i1> [[TOBOOL]], [[TOBOOL1]]
//CHECK:         ret <4 x i1> [[HLSL_OR]]
bool4 test_or_float4(float4 x, float4 y)
{
    return or(x, y);
}

