// RUN: %clang_cc1 -triple powerpc64le -target-feature +altivec -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple powerpc64le -target-feature +altivec -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple powerpc64le -target-feature +altivec -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Unary inc/dec on vector types, covering the CIRGenExprScalar.cpp vector
// branch:
//   * integer vectors go through emitIntIncOrDec -> cir.inc/cir.dec
//   * float vectors go through emitFloatIncOrDec -> cir.fadd/cir.fsub

typedef int vi4 __attribute__((vector_size(16)));
typedef unsigned uvi4 __attribute__((vector_size(16)));
typedef short vsh8 __attribute__((vector_size(16)));
typedef float vf4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));

void vecIntIncDec(vi4 a) {
  ++a;
  --a;
  a++;
  a--;
}
// CIR-LABEL: @_Z12vecIntIncDecDv4_i
// CIR:  cir.inc %{{.+}} : !cir.vector<4 x !s32i>
// CIR:  cir.dec %{{.+}} : !cir.vector<4 x !s32i>
// CIR:  cir.inc %{{.+}} : !cir.vector<4 x !s32i>
// CIR:  cir.dec %{{.+}} : !cir.vector<4 x !s32i>

// LLVM-LABEL: @_Z12vecIntIncDecDv4_i
// LLVM: add <4 x i32> %{{.+}}, splat (i32 1)
// LLVM: sub <4 x i32> %{{.+}}, splat (i32 1)
// LLVM: add <4 x i32> %{{.+}}, splat (i32 1)
// LLVM: sub <4 x i32> %{{.+}}, splat (i32 1)

// OGCG-LABEL: @_Z12vecIntIncDecDv4_i
// OGCG: add <4 x i32> %{{.+}}, splat (i32 1)
// OGCG: add <4 x i32> %{{.+}}, splat (i32 -1)
// OGCG: add <4 x i32> %{{.+}}, splat (i32 1)
// OGCG: add <4 x i32> %{{.+}}, splat (i32 -1)

void vecUIntIncDec(uvi4 b) {
  ++b;
  --b;
  b++;
  b--;
}
// CIR-LABEL: @_Z13vecUIntIncDecDv4_j
// CIR:  cir.inc %{{.+}} : !cir.vector<4 x !u32i>
// CIR:  cir.dec %{{.+}} : !cir.vector<4 x !u32i>
// CIR:  cir.inc %{{.+}} : !cir.vector<4 x !u32i>
// CIR:  cir.dec %{{.+}} : !cir.vector<4 x !u32i>

// LLVM-LABEL: @_Z13vecUIntIncDecDv4_j
// LLVM: add <4 x i32> %{{.+}}, splat (i32 1)
// LLVM: sub <4 x i32> %{{.+}}, splat (i32 1)
// LLVM: add <4 x i32> %{{.+}}, splat (i32 1)
// LLVM: sub <4 x i32> %{{.+}}, splat (i32 1)

// OGCG-LABEL: @_Z13vecUIntIncDecDv4_j
// OGCG: add <4 x i32> %{{.+}}, splat (i32 1)
// OGCG: add <4 x i32> %{{.+}}, splat (i32 -1)
// OGCG: add <4 x i32> %{{.+}}, splat (i32 1)
// OGCG: add <4 x i32> %{{.+}}, splat (i32 -1)

void vecShortIncDec(vsh8 c) {
  ++c;
  --c;
  c++;
  c--;
}
// CIR-LABEL: @_Z14vecShortIncDecDv8_s
// CIR:  cir.inc %{{.+}} : !cir.vector<8 x !s16i>
// CIR:  cir.dec %{{.+}} : !cir.vector<8 x !s16i>
// CIR:  cir.inc %{{.+}} : !cir.vector<8 x !s16i>
// CIR:  cir.dec %{{.+}} : !cir.vector<8 x !s16i>

// LLVM-LABEL: @_Z14vecShortIncDecDv8_s
// LLVM: add <8 x i16> %{{.+}}, splat (i16 1)
// LLVM: sub <8 x i16> %{{.+}}, splat (i16 1)
// LLVM: add <8 x i16> %{{.+}}, splat (i16 1)
// LLVM: sub <8 x i16> %{{.+}}, splat (i16 1)

// OGCG-LABEL: @_Z14vecShortIncDecDv8_s
// OGCG: add <8 x i16> %{{.+}}, splat (i16 1)
// OGCG: add <8 x i16> %{{.+}}, splat (i16 -1)
// OGCG: add <8 x i16> %{{.+}}, splat (i16 1)
// OGCG: add <8 x i16> %{{.+}}, splat (i16 -1)

void vecFloatIncDec(vf4 a) {
  ++a;
  --a;
  a++;
  a--;
}
// CIR-LABEL: @_Z14vecFloatIncDecDv4_f
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.float, !cir.vector<4 x !cir.float>
// CIR:  cir.fadd %{{.+}}, %[[ONEVEC]] : !cir.vector<4 x !cir.float>
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.float, !cir.vector<4 x !cir.float>
// CIR:  cir.fsub %{{.+}}, %[[ONEVEC]] : !cir.vector<4 x !cir.float>
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.float, !cir.vector<4 x !cir.float>
// CIR:  cir.fadd %{{.+}}, %[[ONEVEC]] : !cir.vector<4 x !cir.float>
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.float, !cir.vector<4 x !cir.float>
// CIR:  cir.fsub %{{.+}}, %[[ONEVEC]] : !cir.vector<4 x !cir.float>

// LLVM-LABEL: @_Z14vecFloatIncDecDv4_f
// LLVM: fadd <4 x float> %{{.+}}, splat (float 1.000000e+00)
// LLVM: fsub <4 x float> %{{.+}}, splat (float 1.000000e+00)
// LLVM: fadd <4 x float> %{{.+}}, splat (float 1.000000e+00)
// LLVM: fsub <4 x float> %{{.+}}, splat (float 1.000000e+00)

// OGCG-LABEL: @_Z14vecFloatIncDecDv4_f
// OGCG: fadd <4 x float> %{{.+}}, splat (float 1.000000e+00)
// OGCG: fadd <4 x float> %{{.+}}, splat (float -1.000000e+00)
// OGCG: fadd <4 x float> %{{.+}}, splat (float 1.000000e+00)
// OGCG: fadd <4 x float> %{{.+}}, splat (float -1.000000e+00)

void vecDoubleIncDec(vd2 b) {
  ++b;
  --b;
  b++;
  b--;
}
// CIR-LABEL: @_Z15vecDoubleIncDecDv2_d
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.double, !cir.vector<2 x !cir.double>
// CIR:  cir.fadd %{{.+}}, %[[ONEVEC]] : !cir.vector<2 x !cir.double>
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.double, !cir.vector<2 x !cir.double>
// CIR:  cir.fsub %{{.+}}, %[[ONEVEC]] : !cir.vector<2 x !cir.double>
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.double, !cir.vector<2 x !cir.double>
// CIR:  cir.fadd %{{.+}}, %[[ONEVEC]] : !cir.vector<2 x !cir.double>
// CIR:  %[[ONE:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CIR:  %[[ONEVEC:.*]] = cir.vec.splat %[[ONE]] : !cir.double, !cir.vector<2 x !cir.double>
// CIR:  cir.fsub %{{.+}}, %[[ONEVEC]] : !cir.vector<2 x !cir.double>

// LLVM-LABEL: @_Z15vecDoubleIncDecDv2_d
// LLVM: fadd <2 x double> %{{.+}}, splat (double 1.000000e+00)
// LLVM: fsub <2 x double> %{{.+}}, splat (double 1.000000e+00)
// LLVM: fadd <2 x double> %{{.+}}, splat (double 1.000000e+00)
// LLVM: fsub <2 x double> %{{.+}}, splat (double 1.000000e+00)

// OGCG-LABEL: @_Z15vecDoubleIncDecDv2_d
// OGCG: fadd <2 x double> %{{.+}}, splat (double 1.000000e+00)
// OGCG: fadd <2 x double> %{{.+}}, splat (double -1.000000e+00)
// OGCG: fadd <2 x double> %{{.+}}, splat (double 1.000000e+00)
// OGCG: fadd <2 x double> %{{.+}}, splat (double -1.000000e+00)
