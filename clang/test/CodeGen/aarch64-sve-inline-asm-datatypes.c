// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:   -target-feature +neon -S -O1 -o - -emit-llvm %s | FileCheck %s

// Tests to check that all sve datatypes can be passed in as input operands
// and passed out as output operands.

#define SVINT_TEST(DT, KIND)\
DT func_int_##DT##KIND(DT in)\
{\
  DT out;\
  asm volatile (\
    "ptrue p0.b\n"\
    "mov %[out]." #KIND ", p0/m, %[in]." #KIND "\n"\
    : [out] "=w" (out)\
    : [in] "w" (in)\
    : "p0"\
    );\
  return out;\
}

SVINT_TEST(__SVUint8_t,b);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)
SVINT_TEST(__SVUint8_t,h);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)
SVINT_TEST(__SVUint8_t,s);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)
SVINT_TEST(__SVUint8_t,d);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)

SVINT_TEST(__SVUint16_t,b);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)
SVINT_TEST(__SVUint16_t,h);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)
SVINT_TEST(__SVUint16_t,s);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)
SVINT_TEST(__SVUint16_t,d);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)

SVINT_TEST(__SVUint32_t,b);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)
SVINT_TEST(__SVUint32_t,h);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)
SVINT_TEST(__SVUint32_t,s);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)
SVINT_TEST(__SVUint32_t,d);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)

SVINT_TEST(__SVUint64_t,b);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)
SVINT_TEST(__SVUint64_t,h);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)
SVINT_TEST(__SVUint64_t,s);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)
SVINT_TEST(__SVUint64_t,d);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)

SVINT_TEST(__SVInt8_t,b);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)
SVINT_TEST(__SVInt8_t,h);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)
SVINT_TEST(__SVInt8_t,s);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)
SVINT_TEST(__SVInt8_t,d);
// CHECK: call <vscale x 16 x i8> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 16 x i8> %in)

SVINT_TEST(__SVInt16_t,b);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)
SVINT_TEST(__SVInt16_t,h);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)
SVINT_TEST(__SVInt16_t,s);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)
SVINT_TEST(__SVInt16_t,d);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 8 x i16> %in)

SVINT_TEST(__SVInt32_t,b);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)
SVINT_TEST(__SVInt32_t,h);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)
SVINT_TEST(__SVInt32_t,s);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)
SVINT_TEST(__SVInt32_t,d);
// CHECK: call <vscale x 4 x i32> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in)

SVINT_TEST(__SVInt64_t,b);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)
SVINT_TEST(__SVInt64_t,h);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)
SVINT_TEST(__SVInt64_t,s);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)
SVINT_TEST(__SVInt64_t,d);
// CHECK: call <vscale x 2 x i64> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 2 x i64> %in)


//Test that floats can also be used as datatypes for integer instructions
//and check all the variants which would not be possible with a float
//instruction
SVINT_TEST(__SVFloat16_t,b);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 8 x half> %in)
SVINT_TEST(__SVFloat16_t,h);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 8 x half> %in)
SVINT_TEST(__SVFloat16_t,s);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 8 x half> %in)
SVINT_TEST(__SVFloat16_t,d);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 8 x half> %in)

SVINT_TEST(__SVFloat32_t,b);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in)
SVINT_TEST(__SVFloat32_t,h);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in)
SVINT_TEST(__SVFloat32_t,s);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in)
SVINT_TEST(__SVFloat32_t,d);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in)

SVINT_TEST(__SVFloat64_t,b);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.b\0Amov $0.b, p0/m, $1.b\0A", "=w,w,~{p0}"(<vscale x 2 x double> %in)
SVINT_TEST(__SVFloat64_t,h);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.b\0Amov $0.h, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 2 x double> %in)
SVINT_TEST(__SVFloat64_t,s);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.b\0Amov $0.s, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 2 x double> %in)
SVINT_TEST(__SVFloat64_t,d);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.b\0Amov $0.d, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 2 x double> %in)


#define SVBOOL_TEST(KIND)\
__SVBool_t func_bool_##KIND(__SVBool_t in1, __SVBool_t in2)\
{\
  __SVBool_t out;\
  asm volatile (\
    "zip1 %[out]." #KIND ", %[in1]." #KIND ", %[in2]." #KIND "\n"\
    : [out] "=Upa" (out)\
    :  [in1] "Upa" (in1),\
      [in2] "Upa" (in2)\
    :);\
  return out;\
}

SVBOOL_TEST(b) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "zip1 $0.b, $1.b, $2.b\0A", "=@3Upa,@3Upa,@3Upa"(<vscale x 16 x i1> %in1, <vscale x 16 x i1> %in2)
SVBOOL_TEST(h) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "zip1 $0.h, $1.h, $2.h\0A", "=@3Upa,@3Upa,@3Upa"(<vscale x 16 x i1> %in1, <vscale x 16 x i1> %in2)
SVBOOL_TEST(s) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "zip1 $0.s, $1.s, $2.s\0A", "=@3Upa,@3Upa,@3Upa"(<vscale x 16 x i1> %in1, <vscale x 16 x i1> %in2)
SVBOOL_TEST(d) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "zip1 $0.d, $1.d, $2.d\0A", "=@3Upa,@3Upa,@3Upa"(<vscale x 16 x i1> %in1, <vscale x 16 x i1> %in2)


#define SVBOOL_TEST_UPL(DT, KIND)\
__SVBool_t func_bool_upl_##KIND(__SVBool_t in1, DT in2, DT in3)\
{\
  __SVBool_t out;\
  asm volatile (\
    "fadd %[out]." #KIND ", %[in1]." #KIND ", %[in2]." #KIND ", %[in3]." #KIND "\n"\
    : [out] "=w" (out)\
    :  [in1] "Upl" (in1),\
      [in2] "w" (in2),\
      [in3] "w" (in3)\
    :);\
  return out;\
}

SVBOOL_TEST_UPL(__SVInt8_t, b) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "fadd $0.b, $1.b, $2.b, $3.b\0A", "=w,@3Upl,w,w"(<vscale x 16 x i1> %in1, <vscale x 16 x i8> %in2, <vscale x 16 x i8> %in3)
SVBOOL_TEST_UPL(__SVInt16_t, h) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "fadd $0.h, $1.h, $2.h, $3.h\0A", "=w,@3Upl,w,w"(<vscale x 16 x i1> %in1, <vscale x 8 x i16> %in2, <vscale x 8 x i16> %in3)
SVBOOL_TEST_UPL(__SVInt32_t, s) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "fadd $0.s, $1.s, $2.s, $3.s\0A", "=w,@3Upl,w,w"(<vscale x 16 x i1> %in1, <vscale x 4 x i32> %in2, <vscale x 4 x i32> %in3)
SVBOOL_TEST_UPL(__SVInt64_t, d) ;
// CHECK: call <vscale x 16 x i1> asm sideeffect "fadd $0.d, $1.d, $2.d, $3.d\0A", "=w,@3Upl,w,w"(<vscale x 16 x i1> %in1, <vscale x 2 x i64> %in2, <vscale x 2 x i64> %in3)

#define SVFLOAT_TEST(DT,KIND)\
DT func_float_##DT##KIND(DT inout1, DT in2)\
{\
  asm volatile (\
    "ptrue p0." #KIND ", #1 \n"\
    "fsub %[inout1]." #KIND ", p0/m, %[inout1]." #KIND ", %[in2]." #KIND "\n"\
    : [inout1] "=w" (inout1)\
    : "[inout1]" (inout1),\
      [in2] "w" (in2)\
    : "p0");\
  return inout1 ;\
}\

SVFLOAT_TEST(__SVFloat16_t,s);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.s, #1 \0Afsub $0.s, p0/m, $0.s, $2.s\0A", "=w,0,w,~{p0}"(<vscale x 8 x half> %inout1, <vscale x 8 x half> %in2)
SVFLOAT_TEST(__SVFloat16_t,d);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.d, #1 \0Afsub $0.d, p0/m, $0.d, $2.d\0A", "=w,0,w,~{p0}"(<vscale x 8 x half> %inout1, <vscale x 8 x half> %in2)

SVFLOAT_TEST(__SVFloat32_t,s);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.s, #1 \0Afsub $0.s, p0/m, $0.s, $2.s\0A", "=w,0,w,~{p0}"(<vscale x 4 x float> %inout1, <vscale x 4 x float> %in2)
SVFLOAT_TEST(__SVFloat32_t,d);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.d, #1 \0Afsub $0.d, p0/m, $0.d, $2.d\0A", "=w,0,w,~{p0}"(<vscale x 4 x float> %inout1, <vscale x 4 x float> %in2)

SVFLOAT_TEST(__SVFloat64_t,s);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.s, #1 \0Afsub $0.s, p0/m, $0.s, $2.s\0A", "=w,0,w,~{p0}"(<vscale x 2 x double> %inout1, <vscale x 2 x double> %in2)
SVFLOAT_TEST(__SVFloat64_t,d);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.d, #1 \0Afsub $0.d, p0/m, $0.d, $2.d\0A", "=w,0,w,~{p0}"(<vscale x 2 x double> %inout1, <vscale x 2 x double> %in2)

#define SVFLOAT_TEST_Y(DT, KIND)\
__SVBool_t func_float_y_##KIND(DT in1, DT in2)\
{\
  __SVBool_t out;\
  asm volatile (\
    "fmul %[out]." #KIND ", %[in1]." #KIND ", %[in2]." #KIND "\n"\
    : [out] "=w" (out)\
    :  [in1] "w" (in1),\
      [in2] "y" (in2)\
    :);\
  return out;\
}

SVFLOAT_TEST_Y(__SVFloat16_t,h);
// CHECK: call <vscale x 16 x i1> asm sideeffect "fmul $0.h, $1.h, $2.h\0A", "=w,w,y"(<vscale x 8 x half> %in1, <vscale x 8 x half> %in2)
SVFLOAT_TEST_Y(__SVFloat32_t,s);
// CHECK: call <vscale x 16 x i1> asm sideeffect "fmul $0.s, $1.s, $2.s\0A", "=w,w,y"(<vscale x 4 x float> %in1, <vscale x 4 x float> %in2)
SVFLOAT_TEST_Y(__SVFloat64_t,d);
// CHECK: call <vscale x 16 x i1> asm sideeffect "fmul $0.d, $1.d, $2.d\0A", "=w,w,y"(<vscale x 2 x double> %in1, <vscale x 2 x double> %in2)


// Another test for floats to include h suffix

#define SVFLOAT_CVT_TEST(DT1,KIND1,DT2,KIND2)\
DT1 func_float_cvt_##DT1##KIND1##DT2##KIND2(DT2 in1)\
{\
  DT1 out1 ;\
  asm volatile (\
    "ptrue p0." #KIND2 ", #1 \n"\
    "fcvt %[out1]." #KIND1 ", p0/m, %[in1]." #KIND2 "\n"\
    : [out1] "=w" (out1)\
    : [in1] "w" (in1)\
    : "p0");\
  return out1 ;\
}\

SVFLOAT_CVT_TEST(__SVFloat64_t,d,__SVFloat32_t,s);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.s, #1 \0Afcvt $0.d, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in1)
SVFLOAT_CVT_TEST(__SVFloat64_t,d,__SVFloat16_t,h);
// CHECK: call <vscale x 2 x double> asm sideeffect "ptrue p0.h, #1 \0Afcvt $0.d, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 8 x half> %in1)
SVFLOAT_CVT_TEST(__SVFloat32_t,s,__SVFloat16_t,h);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.h, #1 \0Afcvt $0.s, p0/m, $1.h\0A", "=w,w,~{p0}"(<vscale x 8 x half> %in1)
SVFLOAT_CVT_TEST(__SVFloat32_t,s,__SVFloat64_t,d);
// CHECK: call <vscale x 4 x float> asm sideeffect "ptrue p0.d, #1 \0Afcvt $0.s, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 2 x double> %in1)
SVFLOAT_CVT_TEST(__SVFloat16_t,h,__SVFloat64_t,d);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.d, #1 \0Afcvt $0.h, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 2 x double> %in1)
SVFLOAT_CVT_TEST(__SVFloat16_t,h,__SVFloat32_t,s);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.s, #1 \0Afcvt $0.h, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in1)

//Test a mix of float and ints
SVFLOAT_CVT_TEST(__SVInt16_t,h,__SVFloat32_t,s);
// CHECK: call <vscale x 8 x i16> asm sideeffect "ptrue p0.s, #1 \0Afcvt $0.h, p0/m, $1.s\0A", "=w,w,~{p0}"(<vscale x 4 x float> %in1)
SVFLOAT_CVT_TEST(__SVFloat16_t,s,__SVUint32_t,d);
// CHECK: call <vscale x 8 x half> asm sideeffect "ptrue p0.d, #1 \0Afcvt $0.s, p0/m, $1.d\0A", "=w,w,~{p0}"(<vscale x 4 x i32> %in1)
