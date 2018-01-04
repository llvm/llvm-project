// RUN: c-index-test -test-print-type %s -cl-std=CL2.0 | FileCheck %s

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

// CHECK: VarDecl=scalarHalf:11:8 (Definition){{( \(invalid\))?}} [type=half] [typekind=Half] [isPOD=1]
// CHECK: VarDecl=vectorHalf:12:9 (Definition) [type=half4] [typekind=Typedef] [canonicaltype=half __attribute__((ext_vector_type(4)))] [canonicaltypekind=Unexposed] [isPOD=1]
// CHECK: VarDecl=scalarFloat:13:9 (Definition) [type=float] [typekind=Float] [isPOD=1]
// CHECK: VarDecl=vectorFloat:14:10 (Definition) [type=float4] [typekind=Typedef] [canonicaltype=float __attribute__((ext_vector_type(4)))] [canonicaltypekind=Unexposed] [isPOD=1]
// CHECK: VarDecl=scalarDouble:15:10 (Definition){{( \(invalid\))?}} [type=double] [typekind=Double] [isPOD=1]
// CHECK: VarDecl=vectorDouble:16:11 (Definition) [type=double4] [typekind=Typedef] [canonicaltype=double __attribute__((ext_vector_type(4)))] [canonicaltypekind=Unexposed] [isPOD=1]

#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable

void kernel OCLImage1dROTest(read_only image1d_t scalarOCLImage1dRO);
void kernel OCLImage1dArrayROTest(read_only image1d_array_t scalarOCLImage1dArrayRO);
void kernel OCLImage1dBufferROTest(read_only image1d_buffer_t scalarOCLImage1dBufferRO);
void kernel OCLImage2dROTest(read_only image2d_t scalarOCLImage2dRO);
void kernel OCLImage2dArrayROTest(read_only image2d_array_t scalarOCLImage2dArrayRO);
void kernel OCLImage2dDepthROTest(read_only image2d_depth_t scalarOCLImage2dDepthRO);
void kernel OCLImage2dArrayDepthROTest(read_only image2d_array_depth_t scalarOCLImage2dArrayDepthRO);
void kernel OCLImage2dMSAAROTest(read_only image2d_msaa_t scalarOCLImage2dMSAARO);
void kernel OCLImage2dArrayMSAAROTest(read_only image2d_array_msaa_t scalarOCLImage2dArrayMSAARO);
void kernel OCLImage2dMSAADepthROTest(read_only image2d_msaa_depth_t scalarOCLImage2dMSAADepthRO);
void kernel OCLImage2dArrayMSAADepthROTest(read_only image2d_array_msaa_depth_t scalarOCLImage2dArrayMSAADepthRO);
void kernel OCLImage3dROTest(read_only image3d_t scalarOCLImage3dRO);

// CHECK: ParmDecl=scalarOCLImage1dRO:28:50 (Definition) [type=__read_only image1d_t] [typekind=OCLImage1dRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage1dArrayRO:29:61 (Definition) [type=__read_only image1d_array_t] [typekind=OCLImage1dArrayRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage1dBufferRO:30:63 (Definition) [type=__read_only image1d_buffer_t] [typekind=OCLImage1dBufferRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dRO:31:50 (Definition) [type=__read_only image2d_t] [typekind=OCLImage2dRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayRO:32:61 (Definition) [type=__read_only image2d_array_t] [typekind=OCLImage2dArrayRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dDepthRO:33:61 (Definition) [type=__read_only image2d_depth_t] [typekind=OCLImage2dDepthRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayDepthRO:34:72 (Definition) [type=__read_only image2d_array_depth_t] [typekind=OCLImage2dArrayDepthRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dMSAARO:35:59 (Definition){{( \(invalid\))?}} [type=__read_only image2d_msaa_t] [typekind=OCLImage2dMSAARO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayMSAARO:36:70 (Definition){{( \(invalid\))?}} [type=__read_only image2d_array_msaa_t] [typekind=OCLImage2dArrayMSAARO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dMSAADepthRO:37:70 (Definition){{( \(invalid\))?}} [type=__read_only image2d_msaa_depth_t] [typekind=OCLImage2dMSAADepthRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayMSAADepthRO:38:81 (Definition){{( \(invalid\))?}} [type=__read_only image2d_array_msaa_depth_t] [typekind=OCLImage2dArrayMSAADepthRO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage3dRO:39:50 (Definition) [type=__read_only image3d_t] [typekind=OCLImage3dRO] [isPOD=1]

void kernel OCLImage1dWOTest(write_only image1d_t scalarOCLImage1dWO);
void kernel OCLImage1dArrayWOTest(write_only image1d_array_t scalarOCLImage1dArrayWO);
void kernel OCLImage1dBufferWOTest(write_only image1d_buffer_t scalarOCLImage1dBufferWO);
void kernel OCLImage2dWOTest(write_only image2d_t scalarOCLImage2dWO);
void kernel OCLImage2dArrayWOTest(write_only image2d_array_t scalarOCLImage2dArrayWO);
void kernel OCLImage2dDepthWOTest(write_only image2d_depth_t scalarOCLImage2dDepthWO);
void kernel OCLImage2dArrayDepthWOTest(write_only image2d_array_depth_t scalarOCLImage2dArrayDepthWO);
void kernel OCLImage2dMSAAWOTest(write_only image2d_msaa_t scalarOCLImage2dMSAAWO);
void kernel OCLImage2dArrayMSAAWOTest(write_only image2d_array_msaa_t scalarOCLImage2dArrayMSAAWO);
void kernel OCLImage2dMSAADepthWOTest(write_only image2d_msaa_depth_t scalarOCLImage2dMSAADepthWO);
void kernel OCLImage2dArrayMSAADepthWOTest(write_only image2d_array_msaa_depth_t scalarOCLImage2dArrayMSAADepthWO);
void kernel OCLImage3dWOTest(write_only image3d_t scalarOCLImage3dWO);

// CHECK: ParmDecl=scalarOCLImage1dWO:54:51 (Definition) [type=__write_only image1d_t] [typekind=OCLImage1dWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage1dArrayWO:55:62 (Definition) [type=__write_only image1d_array_t] [typekind=OCLImage1dArrayWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage1dBufferWO:56:64 (Definition) [type=__write_only image1d_buffer_t] [typekind=OCLImage1dBufferWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dWO:57:51 (Definition) [type=__write_only image2d_t] [typekind=OCLImage2dWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayWO:58:62 (Definition) [type=__write_only image2d_array_t] [typekind=OCLImage2dArrayWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dDepthWO:59:62 (Definition) [type=__write_only image2d_depth_t] [typekind=OCLImage2dDepthWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayDepthWO:60:73 (Definition) [type=__write_only image2d_array_depth_t] [typekind=OCLImage2dArrayDepthWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dMSAAWO:61:60 (Definition){{( \(invalid\))?}} [type=__write_only image2d_msaa_t] [typekind=OCLImage2dMSAAWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayMSAAWO:62:71 (Definition){{( \(invalid\))?}} [type=__write_only image2d_array_msaa_t] [typekind=OCLImage2dArrayMSAAWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dMSAADepthWO:63:71 (Definition){{( \(invalid\))?}} [type=__write_only image2d_msaa_depth_t] [typekind=OCLImage2dMSAADepthWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayMSAADepthWO:64:82 (Definition){{( \(invalid\))?}} [type=__write_only image2d_array_msaa_depth_t] [typekind=OCLImage2dArrayMSAADepthWO] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage3dWO:65:51 (Definition){{( \(invalid\))?}} [type=__write_only image3d_t] [typekind=OCLImage3dWO] [isPOD=1]

void kernel OCLImage1dRWTest(read_write image1d_t scalarOCLImage1dRW);
void kernel OCLImage1dArrayRWTest(read_write image1d_array_t scalarOCLImage1dArrayRW);
void kernel OCLImage1dBufferRWTest(read_write image1d_buffer_t scalarOCLImage1dBufferRW);
void kernel OCLImage2dRWTest(read_write image2d_t scalarOCLImage2dRW);
void kernel OCLImage2dArrayRWTest(read_write image2d_array_t scalarOCLImage2dArrayRW);
void kernel OCLImage2dDepthRWTest(read_write image2d_depth_t scalarOCLImage2dDepthRW);
void kernel OCLImage2dArrayDepthRWTest(read_write image2d_array_depth_t scalarOCLImage2dArrayDepthRW);
void kernel OCLImage2dMSAARWTest(read_write image2d_msaa_t scalarOCLImage2dMSAARW);
void kernel OCLImage2dArrayMSAARWTest(read_write image2d_array_msaa_t scalarOCLImage2dArrayMSAARW);
void kernel OCLImage2dMSAADepthRWTest(read_write image2d_msaa_depth_t scalarOCLImage2dMSAADepthRW);
void kernel OCLImage2dArrayMSAADepthRWTest(read_write image2d_array_msaa_depth_t scalarOCLImage2dArrayMSAADepthRW);
void kernel OCLImage3dRWTest(read_write image3d_t scalarOCLImage3dRW);

// CHECK: ParmDecl=scalarOCLImage1dRW:80:51 (Definition) [type=__read_write image1d_t] [typekind=OCLImage1dRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage1dArrayRW:81:62 (Definition) [type=__read_write image1d_array_t] [typekind=OCLImage1dArrayRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage1dBufferRW:82:64 (Definition) [type=__read_write image1d_buffer_t] [typekind=OCLImage1dBufferRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dRW:83:51 (Definition) [type=__read_write image2d_t] [typekind=OCLImage2dRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayRW:84:62 (Definition) [type=__read_write image2d_array_t] [typekind=OCLImage2dArrayRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dDepthRW:85:62 (Definition) [type=__read_write image2d_depth_t] [typekind=OCLImage2dDepthRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayDepthRW:86:73 (Definition) [type=__read_write image2d_array_depth_t] [typekind=OCLImage2dArrayDepthRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dMSAARW:87:60 (Definition){{( \(invalid\))?}} [type=__read_write image2d_msaa_t] [typekind=OCLImage2dMSAARW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayMSAARW:88:71 (Definition){{( \(invalid\))?}} [type=__read_write image2d_array_msaa_t] [typekind=OCLImage2dArrayMSAARW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dMSAADepthRW:89:71 (Definition){{( \(invalid\))?}} [type=__read_write image2d_msaa_depth_t] [typekind=OCLImage2dMSAADepthRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage2dArrayMSAADepthRW:90:82 (Definition){{( \(invalid\))?}} [type=__read_write image2d_array_msaa_depth_t] [typekind=OCLImage2dArrayMSAADepthRW] [isPOD=1]
// CHECK: ParmDecl=scalarOCLImage3dRW:91:51 (Definition) [type=__read_write image3d_t] [typekind=OCLImage3dRW] [isPOD=1]

void kernel intPipeTestRO(read_only pipe int scalarPipe);
void kernel intPipeTestWO(write_only pipe int scalarPipe);

// CHECK: ParmDecl=scalarPipe:106:46 (Definition) [type=read_only pipe int] [typekind=Pipe] [isPOD=0]
// CHECK: ParmDecl=scalarPipe:107:47 (Definition) [type=write_only pipe int] [typekind=Pipe] [isPOD=0]

#define CLK_ADDRESS_CLAMP_TO_EDGE       2
#define CLK_NORMALIZED_COORDS_TRUE      1
#define CLK_FILTER_NEAREST              0x10

void kernel testMiscOpenCLTypes() {
  const sampler_t scalarOCLSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  clk_event_t scalarOCLEvent;
  queue_t scalarOCLQueue;
  reserve_id_t scalarOCLReserveID;
}

// CHECK: VarDecl=scalarOCLSampler:117:19 (Definition) [type=const sampler_t] [typekind=Typedef] const [canonicaltype=const sampler_t] [canonicaltypekind=OCLSampler] [isPOD=1]
// CHECK: VarDecl=scalarOCLEvent:118:15 (Definition) [type=clk_event_t] [typekind=Typedef] [canonicaltype=clk_event_t] [canonicaltypekind=Unexposed] [isPOD=1]
// CHECK: VarDecl=scalarOCLQueue:119:11 (Definition) [type=queue_t] [typekind=Typedef] [canonicaltype=queue_t] [canonicaltypekind=OCLQueue] [isPOD=1]
// CHECK: VarDecl=scalarOCLReserveID:120:16 (Definition) [type=reserve_id_t] [typekind=Typedef] [canonicaltype=reserve_id_t] [canonicaltypekind=OCLReserveID] [isPOD=1]
