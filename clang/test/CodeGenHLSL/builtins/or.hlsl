// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s

//CHECK-LABEL: define noundef i1 @_Z12test_or_boolbb(
//CHECK-SAME: i1 noundef [[X:%.*]], i1 noundef [[Y:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK-NEXT:    [[HLSL_OR:%.*]] = or i1 [[x]], [[y]]
//CHECK-NEXT:    ret i1 [[HLSL_OR]]
//CHECK_NEXT:  }
bool test_or_bool(bool x, bool y)
{
    return or(x, y);

}

//CHECK-LABEL: define noundef <2 x i1> @_Z13test_or_bool2Dv2_bS_(
//CHECK-SAME: <2 x i1> noundef [[X:%.*]], <2 x i1> noundef [[Y:%.*]]) local_unnamed_addr #[[ATTR0]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK-NEXT:    [[HLSL_OR:%.*]] = or <2 xi1> [[x]], [[y]]
//CHECK-NEXT:    ret <2 x i1> [[HLSL_OR]]
//CHECK_NEXT:  }
bool2 test_or_bool2(bool2 x, bool2 y)
{
    return or(x, y);
}

//CHECK-LABEL: define noundef <3 x i1> @_Z13test_or_bool3Dv3_bS_(
//CHECK-SAME: <3 x i1> noundef [[X:%.*]], <3 x i1> noundef [[Y:%.*]]) local_unnamed_addr #[[ATTR0]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK-NEXT:    [[HLSL_OR:%.*]] = or <3 xi1> [[x]], [[y]]
//CHECK-NEXT:    ret <3 x i1> [[HLSL_OR]]
//CHECK_NEXT:  }
bool3 test_or_bool3(bool3 x, bool3 y)
{
    return or(x, y);
}

//CHECK-LABEL: define noundef <4 x i1> @_Z13test_or_bool4Dv4_bS_(
//CHECK-SAME: <4 x i1> noundef [[X:%.*]], <4 x i1> noundef [[Y:%.*]]) local_unnamed_addr #[[ATTR0]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK-NEXT:    [[HLSL_OR:%.*]] = or <4 xi1> [[x]], [[y]]
//CHECK-NEXT:    ret <4 x i1> [[HLSL_OR]]
//CHECK_NEXT:  }
bool4 test_or_bool4(bool4 x, bool4 y)
{
    return or(x, y);
}

//CHECK-LABEL: define noundef i1 @_Z11test_or_intii(
//CHECK-SAME: i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]]) local_unnamed_addr #[[ATTR0]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK_NEXT:    [[0:%.*]] = or i32 [[y]], [[x]]
//CHECK-NEXT:    [[HLSL_OR:%.*]] = icmp ne i32 [[0]], 0
//CHECK-NEXT:    ret i1 [[HLSL_OR]]
//CHECK_NEXT:  }
bool test_or_int(int x, int y)
{
    return or(x, y);
}

//CHECK-LABEL: define noundef <4 x i1> @_Z12test_or_int4Dv4_iS_(
//CHECK-SAME: <4 x i32> noundef [[X:%.*]], <4 x i32> noundef [[Y:%.*]]) local_unnamed_addr #[[ATTR0]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK_NEXT:    [[0:%.*]] = or <4 x i32> [[y]], [[x]]
//CHECK-NEXT:    [[HLSL_OR:%.*]] = icmp ne <4 x i32> [[0]], zeroinitializer
//CHECK-NEXT:    ret <4 x i1> [[HLSL_OR]]
//CHECK_NEXT:  }
bool4 test_or_int4(int4 x, int4 y)
{
    return or(x, y);
}

//CHECK-LABEL: noundef <4 x i1> @_Z14test_or_float4Dv4_fS_(
//CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[X:%.*]], <4 x float> noundef nofpclass(nan inf) [[Y:%.*]]) local_unnamed_addr #[[ATTR0]] {
//CHECK-NEXT:  [[ENTRY:.*:]]
//CHECK-NEXT:    [[TOBOOL:%.*]] = fcmp reassoc nnan ninf nsz arcp afn une <4 x float> [[X]], zeroinitializer
//CHECK-NEXT:    [[TOBOOL1:%.*]] = fcmp reassoc nnan ninf nsz arcp afn une <4 x float> [[Y]], zeroinitializer
//CHECK-NEXT:    [[HLSL_OR:%.*]] = or <4 x i1> [[TOBOOL]], [[TOBOOL1]]
//CHECK-NEXT:    ret <4 x i1> [[HLSL_OR]]
//CHECK_NEXT:  }
bool4 test_or_float4(float4 x, float4 y)
{
    return or(x, y);
}