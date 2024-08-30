// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK

// CHECK: %hlsl.select = select i1
// CHECK: ret i32 %hlsl.select
int test_select_bool_int(bool cond0, int tVal, int fVal) { return select<int>(cond0, tVal, fVal); }

// CHECK: %hlsl.select = select i1
// CHECK: ret <2 x i32> %hlsl.select
vector<int,2> test_select_bool_vector(bool cond0, vector<int, 2> tVal, vector<int, 2> fVal) { return select<vector<int,2> >(cond0, tVal, fVal); }

// CHECK: %4 = extractelement <1 x i1> %extractvec, i32 0
// CHECK: %5 = extractelement <1 x i32> %2, i32 0
// CHECK: %6 = extractelement <1 x i32> %3, i32 0
// CHECK: %7 = select i1 %4, i32 %5, i32 %6
// CHECK: %8 = insertelement <1 x i32> poison, i32 %7, i32 0
// CHECK: ret <1 x i32> %8
vector<int,1> test_select_vector_1(vector<bool,1> cond0, vector<int,1> tVals, vector<int,1> fVals) { return select<int,1>(cond0, tVals, fVals); }

// CHECK: %4 = extractelement <2 x i1> %extractvec, i32 0
// CHECK: %5 = extractelement <2 x i32> %2, i32 0
// CHECK: %6 = extractelement <2 x i32> %3, i32 0
// CHECK: %7 = select i1 %4, i32 %5, i32 %6
// CHECK: %8 = insertelement <2 x i32> poison, i32 %7, i32 0
// CHECK: %9 = extractelement <2 x i1> %extractvec, i32 1
// CHECK: %10 = extractelement <2 x i32> %2, i32 1
// CHECK: %11 = extractelement <2 x i32> %3, i32 1
// CHECK: %12 = select i1 %9, i32 %10, i32 %11
// CHECK: %13 = insertelement <2 x i32> %8, i32 %12, i32 1
// CHECK: ret <2 x i32> %13
vector<int,2> test_select_vector_2(vector<bool, 2> cond0, vector<int, 2> tVals, vector<int, 2> fVals) { return select<int,2>(cond0, tVals, fVals); }

// CHECK: %4 = extractelement <3 x i1> %extractvec, i32 0
// CHECK: %5 = extractelement <3 x i32> %2, i32 0
// CHECK: %6 = extractelement <3 x i32> %3, i32 0
// CHECK: %7 = select i1 %4, i32 %5, i32 %6
// CHECK: %8 = insertelement <3 x i32> poison, i32 %7, i32 0
// CHECK: %9 = extractelement <3 x i1> %extractvec, i32 1
// CHECK: %10 = extractelement <3 x i32> %2, i32 1
// CHECK: %11 = extractelement <3 x i32> %3, i32 1
// CHECK: %12 = select i1 %9, i32 %10, i32 %11
// CHECK: %13 = insertelement <3 x i32> %8, i32 %12, i32 1
// CHECK: %14 = extractelement <3 x i1> %extractvec, i32 2
// CHECK: %15 = extractelement <3 x i32> %2, i32 2
// CHECK: %16 = extractelement <3 x i32> %3, i32 2
// CHECK: %17 = select i1 %14, i32 %15, i32 %16
// CHECK: %18 = insertelement <3 x i32> %13, i32 %17, i32 2
// CHECK: ret <3 x i32> %18
vector<int,3> test_select_vector_3(vector<bool, 3> cond0, vector<int, 3> tVals, vector<int, 3> fVals) { return select<int,3>(cond0, tVals, fVals); }

// CHECK: %4 = extractelement <4 x i1> %extractvec, i32 0
// CHECK: %5 = extractelement <4 x i32> %2, i32 0
// CHECK: %6 = extractelement <4 x i32> %3, i32 0
// CHECK: %7 = select i1 %4, i32 %5, i32 %6
// CHECK: %8 = insertelement <4 x i32> poison, i32 %7, i32 0
// CHECK: %9 = extractelement <4 x i1> %extractvec, i32 1
// CHECK: %10 = extractelement <4 x i32> %2, i32 1
// CHECK: %11 = extractelement <4 x i32> %3, i32 1
// CHECK: %12 = select i1 %9, i32 %10, i32 %11
// CHECK: %13 = insertelement <4 x i32> %8, i32 %12, i32 1
// CHECK: %14 = extractelement <4 x i1> %extractvec, i32 2
// CHECK: %15 = extractelement <4 x i32> %2, i32 2
// CHECK: %16 = extractelement <4 x i32> %3, i32 2
// CHECK: %17 = select i1 %14, i32 %15, i32 %16
// CHECK: %18 = insertelement <4 x i32> %13, i32 %17, i32 2
// CHECK: %19 = extractelement <4 x i1> %extractvec, i32 3
// CHECK: %20 = extractelement <4 x i32> %2, i32 3
// CHECK: %21 = extractelement <4 x i32> %3, i32 3
// CHECK: %22 = select i1 %19, i32 %20, i32 %21
// CHECK: %23 = insertelement <4 x i32> %18, i32 %22, i32 3
// CHECK: ret <4 x i32> %23
vector<int,4> test_select_vector_4(vector<bool, 4> cond0, vector<int, 4> tVals, vector<int, 4> fVals) { return select<int,4>(cond0, tVals, fVals); }



