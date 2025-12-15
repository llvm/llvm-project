// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

//CHECK-LABEL: define hidden noundef <2 x i1> @_Z16test_and_bool2x1u11matrix_typeILm2ELm1EbES_(
//CHECK-SAME: <2 x i1> noundef [[X:%.*]], <2 x i1> noundef [[Y:%.*]])  #[[ATTR0:[0-9]+]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <2 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <2 x i1> [[HLSL_AND_CAST:%.*]]
bool2x1 test_and_bool2x1(bool2x1 x, bool2x1 y)
{
    return and(x, y);
}


//CHECK-LABEL: define hidden noundef <4 x i1> @_Z16test_and_bool2x2u11matrix_typeILm2ELm2EbES_(
//CHECK-SAME: <4 x i1> noundef [[X:%.*]], <4 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <4 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <4 x i1> [[HLSL_AND_CAST:%.*]]
bool2x2 test_and_bool2x2(bool2x2 x, bool2x2 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <6 x i1> @_Z16test_and_bool2x3u11matrix_typeILm2ELm3EbES_(
//CHECK-SAME: <6 x i1> noundef [[X:%.*]], <6 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <6 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <6 x i1> [[HLSL_AND_CAST:%.*]]
bool2x3 test_and_bool2x3(bool2x3 x, bool2x3 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <8 x i1> @_Z16test_and_bool2x4u11matrix_typeILm2ELm4EbES_(
//CHECK-SAME: <8 x i1> noundef [[X:%.*]], <8 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <8 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <8 x i1> [[HLSL_AND_CAST:%.*]]
bool2x4 test_and_bool2x4(bool2x4 x, bool2x4 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <3 x i1> @_Z16test_and_bool3x1u11matrix_typeILm3ELm1EbES_(
//CHECK-SAME: <3 x i1> noundef [[X:%.*]], <3 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <3 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <3 x i1> [[HLSL_AND_CAST:%.*]]
bool3x1 test_and_bool3x1(bool3x1 x, bool3x1 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <6 x i1> @_Z16test_and_bool3x2u11matrix_typeILm3ELm2EbES_(
//CHECK-SAME: <6 x i1> noundef [[X:%.*]], <6 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <6 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <6 x i1> [[HLSL_AND_CAST:%.*]]
bool3x2 test_and_bool3x2(bool3x2 x, bool3x2 y)
{
    return and(x, y);
}


//CHECK-LABEL: define hidden noundef <9 x i1> @_Z16test_and_bool3x3u11matrix_typeILm3ELm3EbES_(
//CHECK-SAME: <9 x i1> noundef [[X:%.*]], <9 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <9 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <9 x i1> [[HLSL_AND_CAST:%.*]]
bool3x3 test_and_bool3x3(bool3x3 x, bool3x3 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <12 x i1> @_Z16test_and_bool3x4u11matrix_typeILm3ELm4EbES_(
//CHECK-SAME: <12 x i1> noundef [[X:%.*]], <12 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <12 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <12 x i1> [[HLSL_AND_CAST:%.*]]
bool3x4 test_and_bool3x4(bool3x4 x, bool3x4 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <4 x i1> @_Z16test_and_bool4x1u11matrix_typeILm4ELm1EbES_(
//CHECK-SAME: <4 x i1> noundef [[X:%.*]], <4 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <4 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <4 x i1> [[HLSL_AND_CAST:%.*]]
bool4x1 test_and_bool4x1(bool4x1 x, bool4x1 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <8 x i1> @_Z16test_and_bool4x2u11matrix_typeILm4ELm2EbES_(
//CHECK-SAME: <8 x i1> noundef [[X:%.*]], <8 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <8 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <8 x i1> [[HLSL_AND_CAST:%.*]]
bool4x2 test_and_bool4x2(bool4x2 x, bool4x2 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <12 x i1> @_Z16test_and_bool4x3u11matrix_typeILm4ELm3EbES_(
//CHECK-SAME: <12 x i1> noundef [[X:%.*]], <12 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <12 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <12 x i1> [[HLSL_AND_CAST:%.*]]
bool4x3 test_and_bool4x3(bool4x3 x, bool4x3 y)
{
    return and(x, y);
}

//CHECK-LABEL: define hidden noundef <16 x i1> @_Z16test_and_bool4x4u11matrix_typeILm4ELm4EbES_(
//CHECK-SAME: <16 x i1> noundef [[X:%.*]], <16 x i1> noundef [[Y:%.*]]) #[[ATTR0]] {
//CHECK-NEXT:  entry:
//CHECK:         [[HLSL_AND:%.*]] = and <16 x i32> [[A:%.*]], [[B:%.*]]
//CHECK:         ret <16 x i1> [[HLSL_AND_CAST:%.*]]
bool4x4 test_and_bool4x4(bool4x4 x, bool4x4 y)
{
    return and(x, y);
}
