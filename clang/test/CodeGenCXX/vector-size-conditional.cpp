// RUN: %clang_cc1 %s -triple x86_64-linux-gnu -Wno-unused -std=c++11 -emit-llvm -o - | FileCheck %s

using FourShorts = short __attribute__((__vector_size__(8)));
using TwoInts = int __attribute__((__vector_size__(8)));
using TwoUInts = unsigned __attribute__((__vector_size__(8)));
using FourInts = int __attribute__((__vector_size__(16)));
using FourUInts = unsigned __attribute__((__vector_size__(16)));
using TwoLongLong = long long __attribute__((__vector_size__(16)));
using FourLongLong = long long __attribute__((__vector_size__(32)));
using TwoFloats = float __attribute__((__vector_size__(8)));
using FourFloats = float __attribute__((__vector_size__(16)));
using TwoDoubles = double __attribute__((__vector_size__(16)));
using FourDoubles = double __attribute__((__vector_size__(32)));

FourShorts four_shorts;
TwoInts two_ints;
TwoUInts two_uints;
FourInts four_ints;
FourUInts four_uints;
TwoLongLong two_ll;
FourLongLong four_ll;
TwoFloats two_floats;
FourFloats four_floats;
TwoDoubles two_doubles;
FourDoubles four_doubles;

short some_short;
unsigned short some_ushort;
int some_int;
float some_float;
unsigned int some_uint;
long long some_ll;
unsigned long long some_ull;
double some_double;

// CHECK: TwoVectorOps
void TwoVectorOps() {
  two_ints ? two_ints : two_ints;
  // CHECK: %[[COND:.+]] = load <2 x i32>
  // CHECK: %[[LHS:.+]] = load <2 x i32>
  // CHECK: %[[RHS:.+]] = load <2 x i32>
  // CHECK: %[[NEZERO:.+]] = icmp ne <2 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <2 x i1> %[[NEZERO]], <2 x i32> %[[LHS]], <2 x i32> %[[RHS]]

  two_ints ? two_floats : two_floats;
  // CHECK: %[[COND:.+]] = load <2 x i32>
  // CHECK: %[[LHS:.+]] = load <2 x float>
  // CHECK: %[[RHS:.+]] = load <2 x float>
  // CHECK: %[[NEZERO:.+]] = icmp ne <2 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <2 x i1> %[[NEZERO]], <2 x float> %[[LHS]], <2 x float> %[[RHS]]

  two_ll ? two_doubles : two_doubles;
  // CHECK: %[[COND:.+]] = load <2 x i64>
  // CHECK: %[[LHS:.+]] = load <2 x double>
  // CHECK: %[[RHS:.+]] = load <2 x double>
  // CHECK: %[[NEZERO:.+]] = icmp ne <2 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <2 x i1> %[[NEZERO]], <2 x double> %[[LHS]], <2 x double> %[[RHS]]
}

// CHECK: TwoScalarOps
void TwoScalarOps() {
  four_shorts ? some_short : some_short;
  // CHECK: %[[COND:.+]] = load <4 x i16>
  // CHECK: %[[LHS:.+]] = load i16
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x i16> poison, i16 %[[LHS]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x i16> %[[LHS_SPLAT_INSERT]], <4 x i16> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load i16
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i16> poison, i16 %[[RHS]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i16> %[[RHS_SPLAT_INSERT]], <4 x i16> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i16> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i16> %[[LHS_SPLAT]], <4 x i16> %[[RHS_SPLAT]]

  four_shorts ? some_ushort : some_ushort;
  // CHECK: %[[COND:.+]] = load <4 x i16>
  // CHECK: %[[LHS:.+]] = load i16
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x i16> poison, i16 %[[LHS]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x i16> %[[LHS_SPLAT_INSERT]], <4 x i16> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load i16
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i16> poison, i16 %[[RHS]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i16> %[[RHS_SPLAT_INSERT]], <4 x i16> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i16> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i16> %[[LHS_SPLAT]], <4 x i16> %[[RHS_SPLAT]]

  four_ints ? some_ushort : some_short;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load i16
  // CHECK: %[[LHS_ZEXT:.+]] = zext i16 %[[LHS]] to i32
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[LHS_ZEXT]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x i32> %[[LHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load i16
  // CHECK: %[[RHS_SEXT:.+]] = sext i16 %[[RHS]] to i32
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[RHS_SEXT]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i32> %[[RHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[LHS_SPLAT]], <4 x i32> %[[RHS_SPLAT]]

  four_ints ? some_int : some_float;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load i32
  // CHECK: %[[LHS_CONV:.+]] = sitofp i32 %[[LHS]] to float
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x float> poison, float %[[LHS_CONV]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x float> %[[LHS_SPLAT_INSERT]], <4 x float> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load float
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x float> poison, float %[[RHS]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x float> %[[RHS_SPLAT_INSERT]], <4 x float> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x float> %[[LHS_SPLAT]], <4 x float> %[[RHS_SPLAT]]

  four_ll ? some_double : some_ll;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load double
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x double> poison, double %[[LHS]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x double> %[[LHS_SPLAT_INSERT]], <4 x double> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load i64
  // CHECK: %[[RHS_CONV:.+]] = sitofp i64 %[[RHS]] to double
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x double> poison, double %[[RHS_CONV]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x double> %[[RHS_SPLAT_INSERT]], <4 x double> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x double> %[[LHS_SPLAT]], <4 x double> %[[RHS_SPLAT]]

  four_ints ? some_int : some_short;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load i32
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[LHS]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x i32> %[[LHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load i16
  // CHECK: %[[RHS_SEXT:.+]] = sext i16 %[[RHS]] to i32
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[RHS_SEXT]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i32> %[[RHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[LHS_SPLAT]], <4 x i32> %[[RHS_SPLAT]]
}

// CHECK: OneScalarOp
void OneScalarOp() {
  four_ints ? four_ints : some_int;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load <4 x i32>
  // CHECK: %[[RHS:.+]] = load i32
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[RHS]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i32> %[[RHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[LHS]], <4 x i32> %[[RHS_SPLAT]]

  four_ints ? four_ints : 5;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load <4 x i32>
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[LHS]], <4 x i32> splat (i32 5)

  four_ints ?: some_float;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[RHS:.+]] = load float
  // CHECK: %[[RHS_CONV:.+]] = fptosi float %[[RHS]] to i32
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[RHS_CONV]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i32> %[[RHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[COND]], <4 x i32> %[[RHS_SPLAT]]

  four_ints ? four_ints : 5.0f;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load <4 x i32>
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[LHS]], <4 x i32> splat (i32 5)

  four_ints ? some_float : four_ints;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load float
  // CHECK: %[[LHS_CONV:.+]] = fptosi float %[[LHS]] to i32
  // CHECK: %[[LHS_SPLAT_INSERT:.+]] = insertelement <4 x i32> poison, i32 %[[LHS_CONV]], i64 0
  // CHECK: %[[LHS_SPLAT:.+]] = shufflevector <4 x i32> %[[LHS_SPLAT_INSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // CHECK: %[[RHS:.+]] = load <4 x i32>
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i32> %[[LHS_SPLAT]], <4 x i32> %[[RHS]]

  four_ints ? four_floats : some_float;
  // CHECK: %[[COND:.+]] = load <4 x i32>
  // CHECK: %[[LHS:.+]] = load <4 x float>
  // CHECK: %[[RHS:.+]] = load float
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x float> poison, float %[[RHS]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x float> %[[RHS_SPLAT_INSERT]], <4 x float> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i32> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x float> %[[LHS]], <4 x float> %[[RHS_SPLAT]]

  four_ll ? four_doubles : 6.0;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load <4 x double>
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x double> %[[LHS]], <4 x double> splat (double 6.{{.+}})

  four_ll ? four_ll : 6.0;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load <4 x i64>
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i64> %[[LHS]], <4 x i64> splat (i64 6)

  four_ll ? four_ll : 6;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load <4 x i64>
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i64> %[[LHS]], <4 x i64> splat (i64 6)

  four_ll ? four_ll : some_int;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load <4 x i64>
  // CHECK: %[[RHS:.+]] = load i32
  // CHECK: %[[RHS_CONV:.+]] = sext i32 %[[RHS]] to i64
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i64> poison, i64 %[[RHS_CONV]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i64> %[[RHS_SPLAT_INSERT]], <4 x i64> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i64> %[[LHS]], <4 x i64> %[[RHS_SPLAT]]

  four_ll ? four_ll : some_ll;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load <4 x i64>
  // CHECK: %[[RHS:.+]] = load i64
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i64> poison, i64 %[[RHS]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i64> %[[RHS_SPLAT_INSERT]], <4 x i64> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i64> %[[LHS]], <4 x i64> %[[RHS_SPLAT]]

  four_ll ? four_ll : some_double;
  // CHECK: %[[COND:.+]] = load <4 x i64>
  // CHECK: %[[LHS:.+]] = load <4 x i64>
  // CHECK: %[[RHS:.+]] = load double
  // CHECK: %[[RHS_CONV:.+]] = fptosi double %[[RHS]] to i64
  // CHECK: %[[RHS_SPLAT_INSERT:.+]] = insertelement <4 x i64> poison, i64 %[[RHS_CONV]], i64 0
  // CHECK: %[[RHS_SPLAT:.+]] = shufflevector <4 x i64> %[[RHS_SPLAT_INSERT]], <4 x i64> poison, <4 x i32> zeroinitializer
  // CHECK: %[[NEZERO:.+]] = icmp ne <4 x i64> %[[COND]], zeroinitializer
  // CHECK: %[[SELECT:.+]] = select <4 x i1> %[[NEZERO]], <4 x i64> %[[LHS]], <4 x i64> %[[RHS_SPLAT]]
}
