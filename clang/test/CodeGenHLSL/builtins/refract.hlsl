// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s --check-prefix=SPVCHECK

// CHECK-LABEL: define noundef nofpclass(nan inf) half @_Z17test_refract_halfDhDhDh(
// CHECK-SAME: half noundef nofpclass(nan inf) [[I:%.*]], half noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ETA]], [[ETA]]
// CHECK-NEXT:    [[TMP0:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[N]], [[I]]
// CHECK-NEXT:    [[TMP1:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[TMP0]], [[TMP0]]
// CHECK-NEXT:    [[SUB1_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[TMP1]]
// CHECK-NEXT:    [[MUL4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL_I]], [[SUB1_I]]
// CHECK-NEXT:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL4_I]]
// CHECK-NEXT:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half [[SUB5_I]], 0xH0000
// CHECK-NEXT:    br i1 [[CMP_I]], label %_ZN4hlsl8__detail12refract_implIDhEET_S2_S2_S2_.exit, label %if.else.i
// CHECK:  if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[MUL6_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ETA]], [[I]]
// CHECK-NEXT:    [[MUL7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[N]], [[I]]
// CHECK-NEXT:    [[MUL8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL7_I]], [[ETA]]
// CHECK-NEXT:    [[TMP2:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.sqrt.f16(half [[SUB5_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn half [[TMP2]], [[MUL8_I]]
// CHECK-NEXT:    [[MUL9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ADD_I]], [[N]]
// CHECK-NEXT:    [[SUB10_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half [[MUL6_I]], [[MUL9_I]]
// CHECK-NEXT:    br label %_ZN4hlsl8__detail12refract_implIDhEET_S2_S2_S2_.exit
// CHECK:  _ZN4hlsl8__detail12refract_implIDhEET_S2_S2_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz half [ [[SUB10_I]], %if.else.i ], [ 0xH0000, %entry ]
// CHECK-NEXT:    ret half [[RETVAL_0_I]]
//
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) half @_Z17test_refract_halfDhDhDh(
// SPVCHECK-SAME: half noundef nofpclass(nan inf) [[I:%.*]], half noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] 
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ETA]], [[ETA]]
// SPVCHECK-NEXT:    [[TMP0:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[N]], [[I]]
// SPVCHECK-NEXT:    [[TMP1:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[TMP0]], [[TMP0]]
// SPVCHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[TMP1]]
// SPVCHECK-NEXT:    [[MUL_4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL_I]], [[SUB_I]]
// SPVCHECK-NEXT:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_4_I]]
// SPVCHECK-NEXT:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half [[SUB5_I]], 0xH0000
// SPVCHECK-NEXT:    br i1 [[CMP_I]], label %_ZN4hlsl8__detail12refract_implIDhEET_S2_S2_S2_.exit, label %if.else.i
// SPVCHECK:  if.else.i:                                        ; preds = %entry
// SPVCHECK-NEXT:    [[MUL_6_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ETA]], [[I]]
// SPVCHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[N]], [[I]]
// SPVCHECK-NEXT:    [[MUL_8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL_7_I]], [[ETA]]
// SPVCHECK-NEXT:    [[TMP2:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.sqrt.f16(half [[SUB5_I]])
// SPVCHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn half [[TMP2]], [[MUL_8_I]]
// SPVCHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ADD_I]], [[N]]
// SPVCHECK-NEXT:    [[SUB10_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half [[MUL_6_I]], [[MUL_9_I]]
// SPVCHECK-NEXT:    br label %_ZN4hlsl8__detail12refract_implIDhEET_S2_S2_S2_.exit
// SPVCHECK:  _ZN4hlsl8__detail12refract_implIDhEET_S2_S2_S2_.exit: ; preds = %entry, %if.else.i
// SPVCHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz half [ [[SUB10_I]], %if.else.i ], [ 0xH0000, %entry ]
// SPVCHECK-NEXT:    ret half [[RETVAL_0_I]]
//
half test_refract_half(half I, half N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z18test_refract_half2Dv2_DhS_Dh(
// CHECK-SAME: <2 x half> noundef nofpclass(nan inf) [[I:%.*]], <2 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[CAST_VTRUNC]], [[ETA]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v2f16(<2 x half> [[N]], <2 x half> [[I]])
// CHECK-NEXT:    [[MUL_2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[HLSL_DOT_I]], [[HLSL_DOT_I]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_2_I]]
// CHECK-NEXT:    [[MUL_3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_3_I]]
// CHECK-NEXT:    [[CAST_VTRUNC_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half [[SUB4_I]], 0xH0000
// CHECK:    br i1 [[CAST_VTRUNC_I]], label %_ZN4hlsl8__detail16refract_vec_implIDhLi2EEEDvT0__T_S3_S3_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[SPLAT_SPLATINSERT_I:%.*]] = insertelement <2 x half> poison, half [[SUB4_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT_I:%.*]] = shufflevector <2 x half> [[SPLAT_SPLATINSERT_I]], <2 x half> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:    [[SPLAT_SPLAT6_I:%.*]] = shufflevector <2 x half> [[ETA]], <2 x half> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> [[SPLAT_SPLAT6_I]], [[I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[HLSL_DOT_I]], [[ETA]]
// CHECK-NEXT:    [[SPLAT_SPLATINSERT10_I:%.*]] = insertelement <2 x half> poison, half [[MUL_9_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT11_I:%.*]] = shufflevector <2 x half> [[SPLAT_SPLATINSERT10_I]], <2 x half> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.sqrt.v2f16(<2 x half> [[SPLAT_SPLAT_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x half> [[TMP0]], [[SPLAT_SPLAT11_I]]
// CHECK-NEXT:    [[MUL_12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x half> [[ADD_I]], [[N]]
// CHECK-NEXT:    [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x half> [[MUL_7_I]], [[MUL_12_I]]
// CHECK-NEXT:    br label %_ZN4hlsl8__detail16refract_vec_implIDhLi2EEEDvT0__T_S3_S3_S2_.exit
// CHECK:  _ZN4hlsl8__detail16refract_vec_implIDhLi2EEEDvT0__T_S3_S3_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz <2 x half> [ %sub13.i, %if.else.i ], [ zeroinitializer, %entry ]
// CHECK-NEXT:    ret <2 x half> [[RETVAL_0_I]]

//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) <2 x half> @_Z18test_refract_half2Dv2_DhS_S_(
// SPVCHECK-SAME: <2 x half> noundef nofpclass(nan inf) [[I:%.*]], <2 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[CONV_I:%.*]] = fpext reassoc nnan ninf nsz arcp afn half [[ETA]] to double
// SPVCHECK-NEXT:    [[SPV_REFRACT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef <2 x half> @llvm.spv.refract.v2f16.f64(<2 x half> [[I]], <2 x half> [[N]], double [[CONV_I]])
// SPVCHECK-NEXT:    ret <2 x half> [[SPV_REFRACT_I]]
//
half2 test_refract_half2(half2 I, half2 N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z18test_refract_half3Dv3_DhS_Dh(
// CHECK-SAME: <3 x half> noundef nofpclass(nan inf) [[I:%.*]], <3 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ETA]], [[ETA]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v3f16(<3 x half> [[N]], <3 x half> [[I]])
// CHECK-NEXT:    [[MUL_2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[HLSL_DOT_I]], [[HLSL_DOT_I]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_2_I]]
// CHECK-NEXT:    [[MUL_3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_3_I]]
// CHECK-NEXT:    [[CAST_VTRUNC_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half [[SUB4_I]], 0xH0000
// CHECK:    br i1 [[CAST_VTRUNC_I]], label %_ZN4hlsl8__detail16refract_vec_implIDhLi3EEEDvT0__T_S3_S3_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[SPLAT_SPLATINSERT_I:%.*]] = insertelement <3 x half> poison, half [[SUB4_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT_I:%.*]] = shufflevector <3 x half> [[SPLAT_SPLATINSERT_I]], <3 x half> poison, <3 x i32> zeroinitializer
// CHECK-NEXT:    [[SPLAT_SPLATINSERT5_I:%.*]] = insertelement <3 x half> poison, half [[ETA]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT6_I:%.*]] = shufflevector <3 x half> [[SPLAT_SPLATINSERT5_I]], <3 x half> poison, <3 x i32> zeroinitializer
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> [[SPLAT_SPLAT6_I]], [[I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[HLSL_DOT_I]], [[ETA]]
// CHECK-NEXT:    [[SPLAT_SPLATINSERT10_I:%.*]] = insertelement <3 x half> poison, half [[MUL_9_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT11_I:%.*]] = shufflevector <3 x half> [[SPLAT_SPLATINSERT10_I]], <3 x half> poison, <3 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.sqrt.v3f16(<3 x half> [[SPLAT_SPLAT_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x half> [[TMP0]], [[SPLAT_SPLAT11_I]]
// CHECK-NEXT:    [[MUL_12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x half> [[ADD_I]], [[N]]
// CHECK-NEXT:   [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x half> [[MUL_7_I]], [[MUL_12_I]]
// CHECK:    br label %_ZN4hlsl8__detail16refract_vec_implIDhLi3EEEDvT0__T_S3_S3_S2_.exit
// CHECK:  _ZN4hlsl8__detail16refract_vec_implIDhLi3EEEDvT0__T_S3_S3_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz <3 x half> [ [[SUB13_I]], %if.else.i ], [ zeroinitializer, %entry ]
// CHECK-NEXT:    ret <3 x half> [[RETVAL_0_I]]
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) <3 x half> @_Z18test_refract_half3Dv3_DhS_Dh(
// SPVCHECK-SAME: <3 x half> noundef nofpclass(nan inf) [[I:%.*]], <3 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[CONV_I:%.*]] = fpext reassoc nnan ninf nsz arcp afn half [[ETA]] to double
// SPVCHECK-NEXT:    [[SPV_REFRACT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef <3 x half> @llvm.spv.refract.v3f16.f64(<3 x half> [[I]], <3 x half> [[N]], double [[CONV_I]])
// SPVCHECK-NEXT:    ret <3 x half> [[SPV_REFRACT_I]]
//
half3 test_refract_half3(half3 I, half3 N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z18test_refract_half4Dv4_DhS_Dh(
// CHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[I:%.*]], <4 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[ETA]], [[ETA]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn half @llvm.dx.fdot.v4f16(<4 x half> [[N]], <4 x half> [[I]])
// CHECK-NEXT:    [[MUL_2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[HLSL_DOT_I]], [[HLSL_DOT_I]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_2_I]]
// CHECK-NEXT:    [[MUL_3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn half 0xH3C00, [[MUL_3_I]]
// CHECK-NEXT:    [[CAST_VTRUNC_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt half [[SUB4_I]], 0xH0000
// CHECK:    br i1 [[CAST_VTRUNC_I]], label %_ZN4hlsl8__detail16refract_vec_implIDhLi4EEEDvT0__T_S3_S3_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[SPLAT_SPLATINSERT_I:%.*]] = insertelement <4 x half> poison, half [[SUB4_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT_I:%.*]] = shufflevector <4 x half> [[SPLAT_SPLATINSERT_I]], <4 x half> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:    [[SPLAT_SPLATINSERT5_I:%.*]] = insertelement <4 x half> poison, half [[ETA]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT6_I:%.*]] = shufflevector <4 x half> [[SPLAT_SPLATINSERT5_I]], <4 x half> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> [[SPLAT_SPLAT6_I]], [[I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn half [[HLSL_DOT_I]], [[ETA]]
// CHECK-NEXT:    [[SPLAT_SPLATINSERT10_I:%.*]] = insertelement <4 x half> poison, half [[MUL_9_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT11_I:%.*]] = shufflevector <4 x half> [[SPLAT_SPLATINSERT10_I]], <4 x half> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.sqrt.v4f16(<4 x half> [[SPLAT_SPLAT_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x half> [[TMP0]], [[SPLAT_SPLAT11_I]]
// CHECK-NEXT:    [[MUL_12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x half> [[ADD_I]], [[N]]
// CHECK-NEXT:   [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x half> [[MUL_7_I]], [[MUL_12_I]]
// CHECK:    br label %_ZN4hlsl8__detail16refract_vec_implIDhLi4EEEDvT0__T_S3_S3_S2_.exit
// CHECK:  _ZN4hlsl8__detail16refract_vec_implIDhLi4EEEDvT0__T_S3_S3_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz <4 x half> [ [[SUB13_I]], %if.else.i ], [ zeroinitializer, %entry ]
// CHECK-NEXT:    ret <4 x half> [[RETVAL_0_I]]
  
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) <4 x half> @_Z18test_refract_half4Dv4_DhS_Dh(
// SPVCHECK-SAME: <4 x half> noundef nofpclass(nan inf) [[I:%.*]], <4 x half> noundef nofpclass(nan inf) [[N:%.*]], half noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[CONV_I:%.*]] = fpext reassoc nnan ninf nsz arcp afn half [[ETA]] to double
// SPVCHECK-NEXT:    [[SPV_REFRACT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef <4 x half> @llvm.spv.refract.v4f16.f64(<4 x half> [[I]], <4 x half> [[N]], double [[CONV_I]])
// SPVCHECK-NEXT:    ret <4 x half> [[SPV_REFRACT_I]]
//
half4 test_refract_half4(half4 I, half4 N, half ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z18test_refract_floatfff(
// CHECK-SAME: float noundef nofpclass(nan inf) [[I:%.*]], float noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[ETA]]
// CHECK-NEXT:    [[TMP0:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[N]], [[I]]
// CHECK-NEXT:    [[TMP1:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[TMP0]], [[TMP0]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[TMP1]]
// CHECK-NEXT:    [[MUL_4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_4_I]]
// CHECK-NEXT:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[SUB5_I]], 0.000000e+00
// CHECK:    br i1 [[CMP_I]], label %_ZN4hlsl8__detail12refract_implIfEET_S2_S2_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[MUL_6_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[I]]
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[N]], [[I]]
// CHECK-NEXT:    [[MUL_8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_7_I]], [[ETA]]
// CHECK-NEXT:    [[TMP2:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(float [[SUB5_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn float [[TMP2]], [[MUL_8_I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ADD_I]], [[N]]
// CHECK-NEXT:    [[SUB10_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float [[MUL_6_I]], [[MUL_9_I]]
// CHECK:    br label %_ZN4hlsl8__detail12refract_implIfEET_S2_S2_S2_.exit
// CHECK: _ZN4hlsl8__detail12refract_implIfEET_S2_S2_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz float [ [[SUB10_I]], %if.else.i ], [ 0.000000e+00, %entry ]
// CHECK-NEXT:    ret float [[RETVAL_0_I]]
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) float @_Z18test_refract_floatfff(
// SPVCHECK-SAME: float noundef nofpclass(nan inf) [[I:%.*]], float noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[ETA]]
// SPVCHECK-NEXT:    [[TMP0:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[N]], [[I]]
// SPVCHECK-NEXT:    [[TMP1:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[TMP0]], [[TMP0]]
// SPVCHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[TMP1]]
// SPVCHECK-NEXT:    [[MUL_4_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_I]], [[SUB_I]]
// SPVCHECK-NEXT:    [[SUB5_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_4_I]]
// SPVCHECK-NEXT:    [[CMP_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[SUB5_I]], 0.000000e+00
// SPVCHECK-NEXT:    br i1 [[CMP_I]], label %_ZN4hlsl8__detail12refract_implIfEET_S2_S2_S2_.exit, label %if.else.i

// SPVCHECK:  if.else.i:                                        ; preds = %entry
// SPVCHECK-NEXT:    [[MUL_6_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[I]]
// SPVCHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[N]], [[I]]
// SPVCHECK-NEXT:    [[MUL_8_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_7_I]], [[ETA]]
// SPVCHECK-NEXT:    [[TMP2:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(float [[SUB5_I]])
// SPVCHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn float [[TMP2]], [[MUL_8_I]]
// SPVCHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ADD_I]], [[N]]
// SPVCHECK-NEXT:    [[SUB10_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float [[MUL_6_I]], [[MUL_9_I]]
// SPVCHECK-NEXT:    br label %_ZN4hlsl8__detail12refract_implIfEET_S2_S2_S2_.exit
// SPVCHECK:  _ZN4hlsl8__detail12refract_implIfEET_S2_S2_S2_.exit: ; preds = %entry, %if.else.i
// SPVCHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz float [ [[SUB10_I]], %if.else.i ], [ 0.000000e+00, %entry ]
// SPVCHECK-NEXT:    ret float [[RETVAL_0_I]]
//
float test_refract_float(float I, float N, float ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z19test_refract_float2Dv2_fS_f(
// CHECK-SAME: <2 x float> noundef nofpclass(nan inf) [[I:%.*]], <2 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[ETA]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v2f32(<2 x float> [[N]], <2 x float> [[I]])
// CHECK-NEXT:    [[MUL_2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[HLSL_DOT_I]], [[HLSL_DOT_I]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_2_I]]
// CHECK-NEXT:    [[MUL_3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_3_I]]
// CHECK-NEXT:    [[CAST_VTRUNC_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[SUB4_I]], 0.000000e+00
// CHECK-NEXT:    br i1 [[CAST_VTRUNC_I]], label %_ZN4hlsl8__detail16refract_vec_implIfLi2EEEDvT0__T_S3_S3_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[SPLAT_SPLATINSERT_I:%.*]] = insertelement <2 x float> poison, float [[SUB4_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT_I:%.*]] = shufflevector <2 x float> [[SPLAT_SPLATINSERT_I]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:    [[SPLAT_SPLATINSERT5_I:%.*]] = insertelement <2 x float> poison, float [[ETA]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT6_I:%.*]] = shufflevector <2 x float> [[SPLAT_SPLATINSERT5_I]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> [[SPLAT_SPLAT6_I]], [[I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[HLSL_DOT_I]], [[ETA]]
// CHECK-NEXT:    [[SPLAT_SPLATINSERT10_I:%.*]] = insertelement <2 x float> poison, float [[MUL_9_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT11_I:%.*]] = shufflevector <2 x float> [[SPLAT_SPLATINSERT10_I]], <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32(<2 x float> [[SPLAT_SPLAT_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <2 x float> [[TMP0]], [[SPLAT_SPLAT11_I]]
// CHECK-NEXT:    [[MUL_12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <2 x float> [[ADD_I]], [[N]]
// CHECK-NEXT:   [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <2 x float> [[MUL_7_I]], [[MUL_12_I]]
// CHECK-NEXT:    br label %_ZN4hlsl8__detail16refract_vec_implIfLi2EEEDvT0__T_S3_S3_S2_.exit
// CHECK:   _ZN4hlsl8__detail16refract_vec_implIfLi2EEEDvT0__T_S3_S3_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz <2 x float> [ [[SUB13_I]], %if.else.i ], [ zeroinitializer, %entry ]
// CHECK-NEXT:    ret <2 x float> [[RETVAL_0_I]]
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) <2 x float> @_Z19test_refract_float2Dv2_fS_f(
// SPVCHECK-SAME: <2 x float> noundef nofpclass(nan inf) [[I:%.*]], <2 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[CONV_I:%.*]] = fpext reassoc nnan ninf nsz arcp afn float [[ETA]] to double
// SPVCHECK-NEXT:    [[SPV_REFRACT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef <2 x float> @llvm.spv.refract.v2f32.f64(<2 x float> [[I]], <2 x float> [[N]], double [[CONV_I]])
// SPVCHECK-NEXT:    ret <2 x float> [[SPV_REFRACT_I]]
//
float2 test_refract_float2(float2 I, float2 N, float ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z19test_refract_float3Dv3_fS_f(
// CHECK-SAME: <3 x float> noundef nofpclass(nan inf) [[I:%.*]], <3 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[ETA]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v3f32(<3 x float> [[N]], <3 x float> [[I]])
// CHECK-NEXT:    [[MUL_2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[HLSL_DOT_I]], [[HLSL_DOT_I]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_2_I]]
// CHECK-NEXT:    [[MUL_3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_3_I]]
// CHECK-NEXT:    [[CAST_VTRUNC_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[SUB4_I]], 0.000000e+00
// CHECK-NEXT:    br i1 [[CAST_VTRUNC_I]], label %_ZN4hlsl8__detail16refract_vec_implIfLi3EEEDvT0__T_S3_S3_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[SPLAT_SPLATINSERT_I:%.*]] = insertelement <3 x float> poison, float [[SUB4_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT_I:%.*]] = shufflevector <3 x float> [[SPLAT_SPLATINSERT_I]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT:    [[SPLAT_SPLATINSERT5_I:%.*]] = insertelement <3 x float> poison, float [[ETA]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT6_I:%.*]] = shufflevector <3 x float> [[SPLAT_SPLATINSERT5_I]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> [[SPLAT_SPLAT6_I]], [[I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[HLSL_DOT_I]], [[ETA]]
// CHECK-NEXT:    [[SPLAT_SPLATINSERT10_I:%.*]] = insertelement <3 x float> poison, float [[MUL_9_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT11_I:%.*]] = shufflevector <3 x float> [[SPLAT_SPLATINSERT10_I]], <3 x float> poison, <3 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32(<3 x float> [[SPLAT_SPLAT_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <3 x float> [[TMP0]], [[SPLAT_SPLAT11_I]]
// CHECK-NEXT:    [[MUL_12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <3 x float> [[ADD_I]], [[N]]
// CHECK-NEXT:   [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <3 x float> [[MUL_7_I]], [[MUL_12_I]]
// CHECK-NEXT:    br label %_ZN4hlsl8__detail16refract_vec_implIfLi3EEEDvT0__T_S3_S3_S2_.exit
// CHECK:   _ZN4hlsl8__detail16refract_vec_implIfLi3EEEDvT0__T_S3_S3_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz <3 x float> [ [[SUB13_I]], %if.else.i ], [ zeroinitializer, %entry ]
// CHECK-NEXT:    ret <3 x float> [[RETVAL_0_I]]
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) <3 x float> @_Z19test_refract_float3Dv3_fS_f(
// SPVCHECK-SAME: <3 x float> noundef nofpclass(nan inf) [[I:%.*]], <3 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[CONV_I:%.*]] = fpext reassoc nnan ninf nsz arcp afn float [[ETA]] to double
// SPVCHECK-NEXT:    [[SPV_REFRACT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef <3 x float> @llvm.spv.refract.v3f32.f64(<3 x float> [[I]], <3 x float> [[N]], double [[CONV_I]])
// SPVCHECK-NEXT:    ret <3 x float> [[SPV_REFRACT_I]]
//
float3 test_refract_float3(float3 I, float3 N, float ETA) {
    return refract(I, N, ETA);
}

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z19test_refract_float4Dv4_fS_f
// CHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[I:%.*]], <4 x float> noundef nofpclass(nan inf) [[N:%.*]], float noundef nofpclass(nan inf) [[ETA:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[MUL_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[ETA]], [[ETA]]
// CHECK-NEXT:    [[HLSL_DOT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn float @llvm.dx.fdot.v4f32(<4 x float> [[N]], <4 x float> [[I]])
// CHECK-NEXT:    [[MUL_2_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[HLSL_DOT_I]], [[HLSL_DOT_I]]
// CHECK-NEXT:    [[SUB_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_2_I]]
// CHECK-NEXT:    [[MUL_3_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[MUL_I]], [[SUB_I]]
// CHECK-NEXT:    [[SUB4_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn float 1.000000e+00, [[MUL_3_I]]
// CHECK-NEXT:    [[CAST_VTRUNC_I:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[SUB4_I]], 0.000000e+00
// CHECK-NEXT:    br i1 [[CAST_VTRUNC_I]], label %_ZN4hlsl8__detail16refract_vec_implIfLi4EEEDvT0__T_S3_S3_S2_.exit, label %if.else.i
// CHECK:   if.else.i:                                        ; preds = %entry
// CHECK-NEXT:    [[SPLAT_SPLATINSERT_I:%.*]] = insertelement <4 x float> poison, float [[SUB4_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT_I:%.*]] = shufflevector <4 x float> [[SPLAT_SPLATINSERT_I]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:    [[SPLAT_SPLATINSERT5_I:%.*]] = insertelement <4 x float> poison, float [[ETA]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT6_I:%.*]] = shufflevector <4 x float> [[SPLAT_SPLATINSERT5_I]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:    [[MUL_7_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> [[SPLAT_SPLAT6_I]], [[I]]
// CHECK-NEXT:    [[MUL_9_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn float [[HLSL_DOT_I]], [[ETA]]
// CHECK-NEXT:    [[SPLAT_SPLATINSERT10_I:%.*]] = insertelement <4 x float> poison, float [[MUL_9_I]], i64 0
// CHECK-NEXT:    [[SPLAT_SPLAT11_I:%.*]] = shufflevector <4 x float> [[SPLAT_SPLATINSERT10_I]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:    [[TMP0:%.*]] = tail call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32(<4 x float> [[SPLAT_SPLAT_I]])
// CHECK-NEXT:    [[ADD_I:%.*]] = fadd reassoc nnan ninf nsz arcp afn <4 x float> [[TMP0]], [[SPLAT_SPLAT11_I]]
// CHECK-NEXT:    [[MUL_12_I:%.*]] = fmul reassoc nnan ninf nsz arcp afn <4 x float> [[ADD_I]], [[N]]
// CHECK-NEXT:   [[SUB13_I:%.*]] = fsub reassoc nnan ninf nsz arcp afn <4 x float> [[MUL_7_I]], [[MUL_12_I]]
// CHECK-NEXT:    br label %_ZN4hlsl8__detail16refract_vec_implIfLi4EEEDvT0__T_S3_S3_S2_.exit
// CHECK:   _ZN4hlsl8__detail16refract_vec_implIfLi4EEEDvT0__T_S3_S3_S2_.exit: ; preds = %entry, %if.else.i
// CHECK-NEXT:    [[RETVAL_0_I:%.*]] = phi nsz <4 x float> [ [[SUB13_I]], %if.else.i ], [ zeroinitializer, %entry ]
// CHECK-NEXT:    ret <4 x float> [[RETVAL_0_I]]
//
// SPVCHECK-LABEL: define spir_func noundef nofpclass(nan inf) <4 x float> @_Z19test_refract_float4Dv4_fS_f(
// SPVCHECK-SAME: <4 x float> noundef nofpclass(nan inf) [[I]], <4 x float> noundef nofpclass(nan inf) [[N]], float noundef nofpclass(nan inf) [[ETA]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SPVCHECK-NEXT:  [[ENTRY:.*:]]
// SPVCHECK-NEXT:    [[CONV_I:%.*]] = fpext reassoc nnan ninf nsz arcp afn float [[ETA]] to double
// SPVCHECK-NEXT:    [[SPV_REFRACT_I:%.*]] = tail call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.spv.refract.v4f32.f64(<4 x float> [[I]], <4 x float> [[N]], double [[CONV_I]])
// SPVCHECK-NEXT:    ret <4 x float> [[SPV_REFRACT_I]]
//
float4 test_refract_float4(float4 I, float4 N, float ETA) {
    return refract(I, N, ETA);
}
