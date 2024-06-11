// RUN: %clang_cc1 -std=c++17 -fclangir -emit-cir -triple x86_64-unknown-linux-gnu %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fclangir -S -emit-llvm -triple x86_64-unknown-linux-gnu %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef int vi4 __attribute__((ext_vector_type(4)));
typedef int vi3 __attribute__((ext_vector_type(3)));
typedef int vi2 __attribute__((ext_vector_type(2)));
typedef double vd2 __attribute__((ext_vector_type(2)));
typedef long vl2 __attribute__((ext_vector_type(2)));
typedef unsigned short vus2 __attribute__((ext_vector_type(2)));

// CIR: cir.func {{@.*vector_int_test.*}}
// LLVM: define void {{@.*vector_int_test.*}}
void vector_int_test(int x) {

  // Vector constant. Not yet implemented. Expected results will change from
  // cir.vec.create to cir.const.
  vi4 a = { 1, 2, 3, 4 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>
  // LLVM: store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %{{[0-9]+}}, align 16

  // Non-const vector initialization.
  vi4 b = { x, 5, 6, x + 1 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>
  // LLVM:      %[[#X1:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#X2:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#SUM:]] = add i32 %[[#X2]], 1
  // LLVM-NEXT: %[[#VEC1:]] = insertelement <4 x i32> undef, i32 %[[#X1]], i64 0
  // LLVM-NEXT: %[[#VEC2:]] = insertelement <4 x i32> %[[#VEC1]], i32 5, i64 1
  // LLVM-NEXT: %[[#VEC3:]] = insertelement <4 x i32> %[[#VEC2]], i32 6, i64 2
  // LLVM-NEXT: %[[#VEC4:]] = insertelement <4 x i32> %[[#VEC3]], i32 %[[#SUM]], i64 3
  // LLVM-NEXT: store <4 x i32> %[[#VEC4]], ptr %{{[0-9]+}}, align 16

  // Incomplete vector initialization.
  vi4 bb = { x, x + 1 };
  // CIR: %[[#zero:]] = cir.const #cir.int<0> : !s32i
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %[[#zero]], %[[#zero]] : !s32i, !s32i, !s32i, !s32i) : !cir.vector<!s32i x 4>
  // LLVM:      %[[#X1:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#X2:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#SUM:]] = add i32 %[[#X2]], 1
  // LLVM-NEXT: %[[#VEC1:]] = insertelement <4 x i32> undef, i32 %[[#X1]], i64 0
  // LLVM-NEXT: %[[#VEC2:]] = insertelement <4 x i32> %[[#VEC1]], i32 %[[#SUM]], i64 1
  // LLVM-NEXT: %[[#VEC3:]] = insertelement <4 x i32> %[[#VEC2]], i32 0, i64 2
  // LLVM-NEXT: %[[#VEC4:]] = insertelement <4 x i32> %[[#VEC3]], i32 0, i64 3
  // LLVM-NEXT: store <4 x i32> %[[#VEC4]], ptr %{{[0-9]+}}, align 16


  // Scalar to vector conversion, a.k.a. vector splat.  Only valid as an
  // operand of a binary operator, not as a regular conversion.
  bb = a + 7;
  // CIR: %[[#seven:]] = cir.const #cir.int<7> : !s32i
  // CIR: %{{[0-9]+}} = cir.vec.splat %[[#seven]] : !s32i, !cir.vector<!s32i x 4>
  // LLVM:      %[[#A:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#BB:]] = add <4 x i32> %[[#A]], <i32 7, i32 7, i32 7, i32 7>
  // LLVM-NEXT: store <4 x i32> %[[#BB]], ptr %{{[0-9]+}}, align 16

  // Vector to vector conversion
  vd2 bbb = { };
  bb = (vi4)bbb;
  // CIR: %{{[0-9]+}} = cir.cast(bitcast, %{{[0-9]+}} : !cir.vector<!cir.double x 2>), !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = bitcast <2 x double> %{{[0-9]+}} to <4 x i32>

  // Extract element
  int c = a[x];
  // CIR: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // LLVM:      %[[#A:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#X:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#EXT:]] = extractelement <4 x i32> %[[#A]], i32 %[[#X]]
  // LLVM-NEXT: store i32 %[[#EXT]], ptr %{{[0-9]+}}, align 4

  // Insert element
  a[x] = x;
  // CIR: %[[#LOADEDVI:]] = cir.load %[[#STORAGEVI:]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR: %[[#UPDATEDVI:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVI]][%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CIR: cir.store %[[#UPDATEDVI]], %[[#STORAGEVI]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>
  // LLVM:      %[[#X1:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#X2:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#A:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#INS:]] = insertelement <4 x i32> %[[#A]], i32 %[[#X1]], i32 %[[#X2]]
  // LLVM-NEXT: store <4 x i32> %[[#INS]], ptr %{{[0-9]+}}, align 16

  // Compound assignment
  a[x] += a[0];
  // CIR: %[[#RHSCA:]] = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CIR: %[[#LHSCA:]] = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // CIR: %[[#SUMCA:]] = cir.binop(add, %[[#LHSCA]], %[[#RHSCA]]) : !s32i
  // CIR: cir.vec.insert %[[#SUMCA]], %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!s32i x 4>
  // LLVM:      %[[#A1:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#RHSCA:]] = extractelement <4 x i32> %[[#A1]], i32 0
  // LLVM-NEXT: %[[#X:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#A2:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#LHSCA:]] = extractelement <4 x i32> %[[#A2]], i32 %[[#X]]
  // LLVM-NEXT: %[[#SUMCA:]] = add i32 %[[#LHSCA]], %[[#RHSCA]]
  // LLVM-NEXT: %[[#A3:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#RES:]] = insertelement <4 x i32> %[[#A3]], i32 %[[#SUMCA]], i32 %[[#X]]
  // LLVM-NEXT: store <4 x i32> %[[#RES]], ptr %{{[0-9]+}}, align 16

  // Binary arithmetic operations
  vi4 d = a + b;
  // CIR: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = add <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 e = a - b;
  // CIR: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = sub <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 f = a * b;
  // CIR: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = mul <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 g = a / b;
  // CIR: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = sdiv <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 h = a % b;
  // CIR: %{{[0-9]+}} = cir.binop(rem, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = srem <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 i = a & b;
  // CIR: %{{[0-9]+}} = cir.binop(and, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = and <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 j = a | b;
  // CIR: %{{[0-9]+}} = cir.binop(or, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = or <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  vi4 k = a ^ b;
  // CIR: %{{[0-9]+}} = cir.binop(xor, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  // LLVM: %{{[0-9]+}} = xor <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}

  // Unary arithmetic operations
  vi4 l = +a;
  // CIR: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM:      %[[#VAL:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: store <4 x i32> %[[#VAL]], ptr %{{[0-9]+}}, align 16
  vi4 m = -a;
  // CIR: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM:      %[[#VAL:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#RES:]] = sub <4 x i32> zeroinitializer, %[[#VAL]]
  // LLVM-NEXT: store <4 x i32> %[[#RES]], ptr %{{[0-9]+}}, align 16
  vi4 n = ~a;
  // CIR: %{{[0-9]+}} = cir.unary(not, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM:      %[[#VAL:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#RES:]] = xor <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, %[[#VAL]]
  // LLVM-NEXT: store <4 x i32> %[[#RES]], ptr %{{[0-9]+}}, align 16

  // TODO: Ternary conditional operator

  // Comparisons
  vi4 o = a == b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM: %[[#RES:]] = icmp eq <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: %[[#EXT:]] = sext <4 x i1> %[[#RES]] to <4 x i32>
  vi4 p = a != b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM: %[[#RES:]] = icmp ne <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: %[[#EXT:]] = sext <4 x i1> %[[#RES]] to <4 x i32>
  vi4 q = a < b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM: %[[#RES:]] = icmp slt <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: %[[#EXT:]] = sext <4 x i1> %[[#RES]] to <4 x i32>
  vi4 r = a > b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM: %[[#RES:]] = icmp sgt <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: %[[#EXT:]] = sext <4 x i1> %[[#RES]] to <4 x i32>
  vi4 s = a <= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM: %[[#RES:]] = icmp sle <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: %[[#EXT:]] = sext <4 x i1> %[[#RES]] to <4 x i32>
  vi4 t = a >= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // LLVM: %[[#RES:]] = icmp sge <4 x i32> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: %[[#EXT:]] = sext <4 x i1> %[[#RES]] to <4 x i32>

  // __builtin_shufflevector
  vi4 u = __builtin_shufflevector(a, b, 7, 5, 3, 1);
  // CIR: %{{[0-9]+}} = cir.vec.shuffle(%{{[0-9]+}}, %{{[0-9]+}} : !cir.vector<!s32i x 4>) [#cir.int<7> : !s64i, #cir.int<5> : !s64i, #cir.int<3> : !s64i, #cir.int<1> : !s64i] : !cir.vector<!s32i x 4>

  // LLVM:      %[[#A:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#B:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHFL:]] = shufflevector <4 x i32> %[[#A]], <4 x i32> %[[#B]], <4 x i32> <i32 7, i32 5, i32 3, i32 1>
  // LLVM-NEXT: store <4 x i32> %[[#SHFL]], ptr %{{[0-9]+}}, align 16

  vi4 v = __builtin_shufflevector(a, b);
  // CIR: %{{[0-9]+}} = cir.vec.shuffle.dynamic %{{[0-9]+}} : !cir.vector<!s32i x 4>, %{{[0-9]+}} : !cir.vector<!s32i x 4>

  // LLVM:      %[[#A:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#B:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#IDXMOD:]] = and <4 x i32> %[[#B]], <i32 3, i32 3, i32 3, i32 3>
  // LLVM-NEXT: %[[#IDX0:]] = extractelement <4 x i32> %[[#IDXMOD]], i64 0
  // LLVM-NEXT: %[[#EXT1:]] = extractelement <4 x i32> %[[#A]], i32 %[[#IDX0]]
  // LLVM-NEXT: %[[#INS1:]] = insertelement <4 x i32> undef, i32 %[[#EXT1]], i64 0
  // LLVM-NEXT: %[[#IDX1:]] = extractelement <4 x i32> %[[#IDXMOD]], i64 1
  // LLVM-NEXT: %[[#EXT2:]] = extractelement <4 x i32> %[[#A]], i32 %[[#IDX1]]
  // LLVM-NEXT: %[[#INS2:]] = insertelement <4 x i32> %[[#INS1]], i32 %[[#EXT2]], i64 1
  // LLVM-NEXT: %[[#IDX2:]] = extractelement <4 x i32> %[[#IDXMOD]], i64 2
  // LLVM-NEXT: %[[#EXT3:]] = extractelement <4 x i32> %[[#A]], i32 %[[#IDX2]]
  // LLVM-NEXT: %[[#INS3:]] = insertelement <4 x i32> %[[#INS2]], i32 %[[#EXT3]], i64 2
  // LLVM-NEXT: %[[#IDX3:]] = extractelement <4 x i32> %[[#IDXMOD]], i64 3
  // LLVM-NEXT: %[[#EXT4:]] = extractelement <4 x i32> %[[#A]], i32 %[[#IDX3]]
  // LLVM-NEXT: %[[#INS4:]] = insertelement <4 x i32> %[[#INS3]], i32 %[[#EXT4]], i64 3
  // LLVM-NEXT: store <4 x i32> %[[#INS4]], ptr %{{[0-9]+}}, align 16
}

// CIR: cir.func {{@.*vector_double_test.*}}
// LLVM: define void {{@.*vector_double_test.*}}
void vector_double_test(int x, double y) {
  // Vector constant. Not yet implemented. Expected results will change from
  // cir.vec.create to cir.const.
  vd2 a = { 1.5, 2.5 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // LLVM: store <2 x double> <double 1.500000e+00, double 2.500000e+00>, ptr %{{[0-9]+}}, align 16

  // Non-const vector initialization.
  vd2 b = { y, y + 1.0 };
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // LLVM:      %[[#Y1:]] = load double, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#Y2:]] = load double, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#SUM:]] = fadd double %[[#Y2]], 1.000000e+00
  // LLVM-NEXT: %[[#VEC1:]] = insertelement <2 x double> undef, double %[[#Y1]], i64 0
  // LLVM-NEXT: %[[#VEC2:]] = insertelement <2 x double> %[[#VEC1]], double %[[#SUM]], i64 1
  // LLVM-NEXT: store <2 x double> %[[#VEC2]], ptr %{{[0-9]+}}, align 16

  // Incomplete vector initialization
  vd2 bb = { y };
  // CIR: [[#dzero:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
  // CIR: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %[[#dzero]] : !cir.double, !cir.double) : !cir.vector<!cir.double x 2>

  // LLVM:      %[[#Y1:]] = load double, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#VEC1:]] = insertelement <2 x double> undef, double %[[#Y1]], i64 0
  // LLVM-NEXT: %[[#VEC2:]] = insertelement <2 x double> %[[#VEC1]], double 0.000000e+00, i64 1
  // LLVM-NEXT: store <2 x double> %[[#VEC2]], ptr %{{[0-9]+}}, align 16

  // Scalar to vector conversion, a.k.a. vector splat.  Only valid as an
  // operand of a binary operator, not as a regular conversion.
  bb = a + 2.5;
  // CIR: %[[#twohalf:]] = cir.const #cir.fp<2.500000e+00> : !cir.double
  // CIR: %{{[0-9]+}} = cir.vec.splat %[[#twohalf]] : !cir.double, !cir.vector<!cir.double x 2>

  // LLVM:      %[[#A:]] = load <2 x double>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#BB:]] = fadd <2 x double> %[[#A]], <double 2.500000e+00, double 2.500000e+00>
  // LLVM-NEXT: store <2 x double> %[[#BB]], ptr %{{[0-9]+}}, align 16

  // Extract element
  double c = a[x];
  // CIR: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : !cir.vector<!cir.double x 2>
  // LLVM: %{{[0-9]+}} = extractelement <2 x double> %{{[0-9]+}}, i32 %{{[0-9]+}}

  // Insert element
  a[x] = y;
  // CIR: %[[#LOADEDVF:]] = cir.load %[[#STORAGEVF:]] : !cir.ptr<!cir.vector<!cir.double x 2>>, !cir.vector<!cir.double x 2>
  // CIR: %[[#UPDATEDVF:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVF]][%{{[0-9]+}} : !s32i] : !cir.vector<!cir.double x 2>
  // CIR: cir.store %[[#UPDATEDVF]], %[[#STORAGEVF]] : !cir.vector<!cir.double x 2>, !cir.ptr<!cir.vector<!cir.double x 2>>

  // LLVM:      %[[#Y:]] = load double, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#X:]] = load i32, ptr %{{[0-9]+}}, align 4
  // LLVM-NEXT: %[[#A:]] = load <2 x double>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#INS:]] = insertelement <2 x double> %[[#A]], double %[[#Y]], i32 %[[#X]]
  // LLVM-NEXT: store <2 x double> %[[#INS]], ptr %{{[0-9]+}}, align 16

  // Binary arithmetic operations
  vd2 d = a + b;
  // CIR: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  // LLVM: %{{[0-9]+}} = fadd <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  vd2 e = a - b;
  // CIR: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  // LLVM: %{{[0-9]+}} = fsub <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  vd2 f = a * b;
  // CIR: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  // LLVM: %{{[0-9]+}} = fmul <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  vd2 g = a / b;
  // CIR: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  // LLVM: %{{[0-9]+}} = fdiv <2 x double> %{{[0-9]+}}, %{{[0-9]+}}

  // Unary arithmetic operations
  vd2 l = +a;
  // CIR: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>
  // LLVM:      %[[#VAL:]] = load <2 x double>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: store <2 x double> %[[#VAL]], ptr %{{[0-9]+}}, align 16
  vd2 m = -a;
  // CIR: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>
  // LLVM:      %[[#VAL:]] = load <2 x double>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#RES:]] = fneg <2 x double> %[[#VAL]]
  // LLVM-NEXT: store <2 x double> %[[#RES]], ptr %{{[0-9]+}}, align 16

  // Comparisons
  vl2 o = a == b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  // LLVM: %[[#RES:]] = fcmp oeq <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: sext <2 x i1> %[[#RES:]] to <2 x i64>
  vl2 p = a != b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  // LLVM: %[[#RES:]] = fcmp une <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: sext <2 x i1> %[[#RES:]] to <2 x i64>
  vl2 q = a < b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  // LLVM: %[[#RES:]] = fcmp olt <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: sext <2 x i1> %[[#RES:]] to <2 x i64>
  vl2 r = a > b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  // LLVM: %[[#RES:]] = fcmp ogt <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: sext <2 x i1> %[[#RES:]] to <2 x i64>
  vl2 s = a <= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  // LLVM: %[[#RES:]] = fcmp ole <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: sext <2 x i1> %[[#RES:]] to <2 x i64>
  vl2 t = a >= b;
  // CIR: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!s64i x 2>
  // LLVM: %[[#RES:]] = fcmp oge <2 x double> %{{[0-9]+}}, %{{[0-9]+}}
  // LLVM-NEXT: sext <2 x i1> %[[#RES:]] to <2 x i64>

  // __builtin_convertvector
  vus2 w = __builtin_convertvector(a, vus2);
  // CIR: %{{[0-9]+}} = cir.cast(float_to_int, %{{[0-9]+}} : !cir.vector<!cir.double x 2>), !cir.vector<!u16i x 2>
  // LLVM: %{{[0-9]+}} = fptoui <2 x double> %{{[0-9]+}} to <2 x i16>
}

// CIR: cir.func {{@.*test_load.*}}
// LLVM: define void {{@.*test_load.*}}
void test_load() {
  vi4 a = { 1, 2, 3, 4 };

  vi2 b;

  b = a.wz;
  // CIR:      %[[#LOAD1:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE1:]] = cir.vec.shuffle(%[[#LOAD1]], %[[#LOAD1]] : !cir.vector<!s32i x 4>) [#cir.int<3> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s32i x 2>
  // CIR-NEXT: cir.store %[[#SHUFFLE1]], %{{[0-9]+}} : !cir.vector<!s32i x 2>, !cir.ptr<!cir.vector<!s32i x 2>>

  // LLVM:      %[[#LOAD1:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE1:]] = shufflevector <4 x i32> %[[#LOAD1]], <4 x i32> %[[#LOAD1]], <2 x i32> <i32 3, i32 2>
  // LLVM-NEXT: store <2 x i32> %[[#SHUFFLE1]], ptr %{{[0-9]+}}, align 8

  int one_elem_load = a.s2;
  // CIR-NEXT: %[[#LOAD8:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#EXTRACT_INDEX:]] = cir.const #cir.int<2> : !s64i
  // CIR-NEXT: %[[#EXTRACT1:]] = cir.vec.extract %[[#LOAD8]][%[[#EXTRACT_INDEX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#EXTRACT1]], %{{[0-9]+}} : !s32i, !cir.ptr<!s32i>

  // LLVM-NEXT: %[[#LOAD8:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#EXTRACT1:]] = extractelement <4 x i32> %[[#LOAD8]], i64 2
  // LLVM-NEXT: store i32 %[[#EXTRACT1]], ptr %{{[0-9]+}}, align 4

}

// CIR: cir.func {{@.*test_store.*}}
// LLVM: define void {{@.*test_store.*}}
void test_store() {
  vi4 a;
  // CIR: %[[#PVECA:]] = cir.alloca !cir.vector<!s32i x 4>
  // LLVM: %[[#PVECA:]] = alloca <4 x i32>

  vi2 b = {1, 2};
  // CIR-NEXT: %[[#PVECB:]] = cir.alloca !cir.vector<!s32i x 2>
  // LLVM-NEXT: %[[#PVECB:]] = alloca <2 x i32>

  vi3 c = {};
  // CIR-NEXT: %[[#PVECC:]] = cir.alloca !cir.vector<!s32i x 3>
  // LLVM-NEXT: %[[#PVECC:]] = alloca <3 x i32>

  a.xy = b;
  // CIR:      %[[#LOAD4RHS:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 2>>, !cir.vector<!s32i x 2>
  // CIR-NEXT: %[[#LOAD5LHS:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE5:]] = cir.vec.shuffle(%[[#LOAD4RHS]], %[[#LOAD4RHS]] : !cir.vector<!s32i x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SHUFFLE6:]] = cir.vec.shuffle(%[[#LOAD5LHS]], %[[#SHUFFLE5]] : !cir.vector<!s32i x 4>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#SHUFFLE6]], %{{[0-9]+}} : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM:      %[[#LOAD4RHS:]] = load <2 x i32>, ptr %{{[0-9]+}}, align 8
  // LLVM-NEXT: %[[#LOAD5LHS:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SHUFFLE5:]] = shufflevector <2 x i32> %[[#LOAD4RHS]], <2 x i32> %[[#LOAD4RHS]], <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // LLVM-NEXT: %[[#SHUFFLE6:]] = shufflevector <4 x i32> %[[#LOAD5LHS]], <4 x i32> %[[#SHUFFLE5]], <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  // LLVM-NEXT: store <4 x i32> %[[#SHUFFLE6]], ptr %{{[0-9]+}}, align 16

  // load single element
  a.s0 = 1;
  // CIR-NEXT: cir.const #cir.int<1>
  // CIR-NEXT: %[[#LOAD7:]] = cir.load %{{[0-9]+}} : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#INSERT_INDEX:]] = cir.const #cir.int<0> : !s64i
  // CIR-NEXT: %[[#INSERT1:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOAD7]][%[[#INSERT_INDEX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#INSERT1]], %{{[0-9]+}} : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#LOAD7:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#INSERT1:]] = insertelement <4 x i32> %[[#LOAD7]], i32 1, i64 0
  // LLVM-NEXT: store <4 x i32> %[[#INSERT1]], ptr %{{[0-9]+}}, align 16

  // extend length from 2 to 4, then merge two vectors
  a.lo = b;
  // CIR:      %[[#VECB:]] = cir.load %[[#PVECB]] : !cir.ptr<!cir.vector<!s32i x 2>>, !cir.vector<!s32i x 2>
  // CIR-NEXT: %[[#VECA:]] = cir.load %[[#PVECA]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#EXTVECB:]] = cir.vec.shuffle(%[[#VECB]], %[[#VECB]] : !cir.vector<!s32i x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.shuffle(%[[#VECA]], %[[#EXTVECB]] : !cir.vector<!s32i x 4>) [#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#PVECA]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM:      %[[#VECB:]] = load <2 x i32>, ptr %[[#PVECB]], align 8
  // LLVM-NEXT: %[[#VECA:]] = load <4 x i32>, ptr %[[#PVECA]], align 16
  // LLVM-NEXT: %[[#EXTVECB:]] = shufflevector <2 x i32> %[[#VECB]], <2 x i32> %[[#VECB]], <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  // LLVM-NEXT: %[[#RESULT:]] = shufflevector <4 x i32> %[[#VECA]], <4 x i32> %[[#EXTVECB]], <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  // LLVM-NEXT: store <4 x i32> %[[#RESULT]], ptr %[[#PVECA]], align 16

  // OpenCL C Specification 6.3.7. Vector Components
  // The suffixes .lo (or .even) and .hi (or .odd) for a 3-component vector type
  // operate as if the 3-component vector type is a 4-component vector type with
  // the value in the w component undefined.
  b = c.hi;

  // CIR-NEXT: %[[#VECC:]] = cir.load %[[#PVECC]] : !cir.ptr<!cir.vector<!s32i x 3>>, !cir.vector<!s32i x 3>
  // CIR-NEXT: %[[#HIPART:]] = cir.vec.shuffle(%[[#VECC]], %[[#VECC]] : !cir.vector<!s32i x 3>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 2>
  // CIR-NEXT: cir.store %[[#HIPART]], %[[#PVECB]] : !cir.vector<!s32i x 2>, !cir.ptr<!cir.vector<!s32i x 2>>

  // LLVM-NEXT: %[[#VECC:]] = load <3 x i32>, ptr %[[#PVECC]], align 16
  // LLVM-NEXT: %[[#HIPART:]] = shufflevector <3 x i32> %[[#VECC]], <3 x i32> %[[#VECC]], <2 x i32> <i32 2, i32 3>
  // LLVM-NEXT: store <2 x i32> %[[#HIPART]], ptr %[[#PVECB]], align 8

  // c.hi is c[2, 3], in which 3 should be ignored in CIRGen for store
  c.hi = b;

  // CIR-NEXT: %[[#VECB:]] = cir.load %[[#PVECB]] : !cir.ptr<!cir.vector<!s32i x 2>>, !cir.vector<!s32i x 2>
  // CIR-NEXT: %[[#VECC:]] = cir.load %[[#PVECC]] : !cir.ptr<!cir.vector<!s32i x 3>>, !cir.vector<!s32i x 3>
  // CIR-NEXT: %[[#EXTVECB:]] = cir.vec.shuffle(%[[#VECB]], %[[#VECB]] : !cir.vector<!s32i x 2>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 3>
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.shuffle(%[[#VECC]], %[[#EXTVECB]] : !cir.vector<!s32i x 3>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<3> : !s32i] : !cir.vector<!s32i x 3>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#PVECC]] : !cir.vector<!s32i x 3>, !cir.ptr<!cir.vector<!s32i x 3>>

  // LLVM-NEXT: %[[#VECB:]] = load <2 x i32>, ptr %[[#PVECB]], align 8
  // LLVM-NEXT: %[[#VECC:]] = load <3 x i32>, ptr %[[#PVECC]], align 16
  // LLVM-NEXT: %[[#EXTVECB:]] = shufflevector <2 x i32> %[[#VECB]], <2 x i32> %[[#VECB]], <3 x i32> <i32 0, i32 1, i32 poison>
  // LLVM-NEXT: %[[#RESULT:]] = shufflevector <3 x i32> %[[#VECC]], <3 x i32> %[[#EXTVECB]], <3 x i32> <i32 0, i32 1, i32 3>
  // LLVM-NEXT: store <3 x i32> %[[#RESULT]], ptr %[[#PVECC]], align 16

}

// CIR: cir.func {{@.*test_build_lvalue.*}}
// LLVM: define void {{@.*test_build_lvalue.*}}
void test_build_lvalue() {
  // special cases only

  vi4 *pv, v;

  // CIR-NEXT: %[[#ALLOCAPV:]] = cir.alloca !cir.ptr<!cir.vector<!s32i x 4>>, !cir.ptr<!cir.ptr<!cir.vector<!s32i x 4>>>, ["pv"] {alignment = 8 : i64}
  // CIR-NEXT: %[[#ALLOCAV:]] = cir.alloca !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>, ["v"] {alignment = 16 : i64}
  // CIR-NEXT: %[[#ALLOCAS:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["s", init] {alignment = 4 : i64}
  // CIR-NEXT: %[[#ALLOCATMP:]] = cir.alloca !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>, ["tmp"] {alignment = 16 : i64}
  // CIR-NEXT: %[[#ALLOCAR:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init] {alignment = 4 : i64}

  // LLVM-NEXT: %[[#ALLOCAPV:]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT: %[[#ALLOCAV:]] = alloca <4 x i32>, i64 1, align 16
  // LLVM-NEXT: %[[#ALLOCAS:]] = alloca i32, i64 1, align 4
  // LLVM-NEXT: %[[#ALLOCATMP:]] = alloca <4 x i32>, i64 1, align 16
  // LLVM-NEXT: %[[#ALLOCAR:]] = alloca i32, i64 1, align 4

  pv->x = 99;
  // CIR-NEXT: %[[#VAL:]] = cir.const #cir.int<99> : !s32i
  // CIR-NEXT: %[[#PV:]] = cir.load %[[#ALLOCAPV]] : !cir.ptr<!cir.ptr<!cir.vector<!s32i x 4>>>, !cir.ptr<!cir.vector<!s32i x 4>>
  // CIR-NEXT: %[[#V:]] = cir.load %[[#PV]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#IDX:]] = cir.const #cir.int<0> : !s64i
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.insert %[[#VAL]], %[[#V]][%[[#IDX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#PV]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#PV:]] = load ptr, ptr %[[#ALLOCAPV]], align 8
  // LLVM-NEXT: %[[#V:]] = load <4 x i32>, ptr %[[#PV]], align 16
  // LLVM-NEXT: %[[#RESULT:]] = insertelement <4 x i32> %[[#V]], i32 99, i64 0
  // LLVM-NEXT: store <4 x i32> %[[#RESULT]], ptr %[[#PV]], align 16

  int s = (v+v).x;

  // CIR-NEXT: %[[#LOAD1:]] = cir.load %[[#ALLOCAV]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#LOAD2:]] = cir.load %[[#ALLOCAV]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#SUM:]] = cir.binop(add, %[[#LOAD1]], %[[#LOAD2]]) : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#SUM]], %[[#ALLOCATMP]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>
  // CIR-NEXT: %[[#TMP:]] = cir.load %[[#ALLOCATMP]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#IDX:]] = cir.const #cir.int<0> : !s64i
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.extract %[[#TMP]][%[[#IDX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#ALLOCAS]] : !s32i, !cir.ptr<!s32i>

  // LLVM-NEXT: %[[#LOAD1:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#LOAD2:]] = load <4 x i32>, ptr %{{[0-9]+}}, align 16
  // LLVM-NEXT: %[[#SUM:]] = add <4 x i32> %[[#LOAD1]], %[[#LOAD2]]
  // LLVM-NEXT: store <4 x i32> %[[#SUM]], ptr %[[#ALLOCATMP]], align 16
  // LLVM-NEXT: %[[#TMP:]] = load <4 x i32>, ptr %[[#ALLOCATMP]], align 16
  // LLVM-NEXT: %[[#RESULT:]] = extractelement <4 x i32> %[[#TMP]], i64 0
  // LLVM-NEXT: store i32 %[[#RESULT]], ptr %[[#ALLOCAS]], align 4

  int r = v.xy.x;
  // CIR-NEXT: %[[#V:]] = cir.load %[[#ALLOCAV]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#IDX:]] = cir.const #cir.int<0> : !s64i
  // CIR-NEXT: %[[#RESULT:]] = cir.vec.extract %[[#V]][%[[#IDX]] : !s64i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: cir.store %[[#RESULT]], %[[#ALLOCAR]] : !s32i, !cir.ptr<!s32i>

  // LLVM-NEXT: %[[#V:]] = load <4 x i32>, ptr %[[#ALLOCAV]], align 16
  // LLVM-NEXT: %[[#RESULT:]] = extractelement <4 x i32> %[[#V]], i64 0
  // LLVM-NEXT: store i32 %[[#RESULT]], ptr %[[#ALLOCAR]], align 4

}

// CIR: cir.func {{@.*test_vec3.*}}
// LLVM: define void {{@.*test_vec3.*}}
void test_vec3() {
  vi3 v = {};
  // CIR-NEXT: %[[#PV:]] = cir.alloca !cir.vector<!s32i x 3>, !cir.ptr<!cir.vector<!s32i x 3>>, ["v", init] {alignment = 16 : i64}
  // CIR:      %[[#VEC4:]] = cir.vec.shuffle(%{{[0-9]+}}, %{{[0-9]+}} : !cir.vector<!s32i x 3>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#PV4:]] = cir.cast(bitcast, %[[#PV]] : !cir.ptr<!cir.vector<!s32i x 3>>), !cir.ptr<!cir.vector<!s32i x 4>>
  // CIR-NEXT: cir.store %[[#VEC4]], %[[#PV4]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-NEXT: %[[#PV:]] = alloca <3 x i32>, i64 1, align 16
  // LLVM-NEXT: store <4 x i32> <i32 0, i32 0, i32 0, i32 undef>, ptr %[[#PV]], align 16

  v + 1;
  // CIR-NEXT: %[[#PV4:]] = cir.cast(bitcast, %[[#PV]] : !cir.ptr<!cir.vector<!s32i x 3>>), !cir.ptr<!cir.vector<!s32i x 4>>
  // CIR-NEXT: %[[#V4:]] = cir.load %[[#PV4]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CIR-NEXT: %[[#V3:]] = cir.vec.shuffle(%[[#V4]], %[[#V4]] : !cir.vector<!s32i x 4>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i] : !cir.vector<!s32i x 3>
  // CIR:      %[[#RES:]] = cir.binop(add, %[[#V3]], %{{[0-9]+}}) : !cir.vector<!s32i x 3>

  // LLVM-NEXT: %[[#V4:]] = load <4 x i32>, ptr %[[#PV:]], align 16
  // LLVM-NEXT: %[[#V3:]] = shufflevector <4 x i32> %[[#V4]], <4 x i32> %[[#V4]], <3 x i32> <i32 0, i32 1, i32 2>
  // LLVM-NEXT: %[[#RES:]] = add <3 x i32> %[[#V3]], <i32 1, i32 1, i32 1>

}
