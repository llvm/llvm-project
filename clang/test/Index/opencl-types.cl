// RUN: c-index-test -test-print-type %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef half half4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef double double4 __attribute__((ext_vector_type(4)));

void kernel testFloatTypes() {
  half scalarHalf;
  half4 vectorHalf;
  float scalarFloat;
  float4 vectorFloat;
  double scalarDouble;
  double4 vectorDouble;
}

// CHECK: VarDecl=scalarHalf:11:8 (Definition) [type=half] [typekind=Half] [isPOD=1]
// CHECK: VarDecl=vectorHalf:12:9 (Definition) [type=half4] [typekind=Typedef] [canonicaltype=half __attribute__((ext_vector_type(4)))] [canonicaltypekind=Unexposed] [isPOD=1]
// CHECK: VarDecl=scalarFloat:13:9 (Definition) [type=float] [typekind=Float] [isPOD=1]
// CHECK: VarDecl=vectorFloat:14:10 (Definition) [type=float4] [typekind=Typedef] [canonicaltype=float __attribute__((ext_vector_type(4)))] [canonicaltypekind=Unexposed] [isPOD=1]
// CHECK: VarDecl=scalarDouble:15:10 (Definition) [type=double] [typekind=Double] [isPOD=1]
// CHECK: VarDecl=vectorDouble:16:11 (Definition) [type=double4] [typekind=Typedef] [canonicaltype=double __attribute__((ext_vector_type(4)))] [canonicaltypekind=Unexposed] [isPOD=1]
