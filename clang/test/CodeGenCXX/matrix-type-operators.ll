; ModuleID = '../clang/test/CodeGenCXX/matrix-type-operators.cpp'
source_filename = "../clang/test/CodeGenCXX/matrix-type-operators.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%struct.identmatrix_t = type { i8 }
%struct.MyMatrix = type { [10 x float] }
%struct.DoubleWrapper1 = type { i32 }
%struct.MyMatrix.0 = type { [90 x double] }
%struct.DoubleWrapper2 = type { i32 }
%struct.IntWrapper = type { i8 }
%struct.MyMatrix.1 = type { [4 x float] }
%struct.MyMatrix.2 = type { [10 x float] }
%struct.MyMatrix.3 = type { [4 x i32] }
%struct.MyMatrix.4 = type { [24 x float] }
%struct.MyMatrix.5 = type { [4 x i32] }
%struct.UnsignedWrapper = type { i8 }

@_ZL11identmatrix = internal constant %struct.identmatrix_t zeroinitializer, align 1
@_ZZ16matrix_subscriptIiiEDTixixfp_fp0_fp1_Eu11matrix_typeILm4ELm4EdET_T0_E1d = linkonce_odr global double 0.000000e+00, align 8

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z17test_add_templatev() #0 {
entry:
  %Mat1 = alloca %struct.MyMatrix, align 4
  %Mat2 = alloca %struct.MyMatrix, align 4
  %call = call noundef <10 x float> @_Z3addIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(ptr noundef nonnull align 4 dereferenceable(40) %Mat1, ptr noundef nonnull align 4 dereferenceable(40) %Mat2)
  %value = getelementptr inbounds nuw %struct.MyMatrix, ptr %Mat1, i32 0, i32 0
  store <10 x float> %call, ptr %value, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef <10 x float> @_Z3addIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(ptr noundef nonnull align 4 dereferenceable(40) %A, ptr noundef nonnull align 4 dereferenceable(40) %B) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  %0 = load ptr, ptr %A.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix, ptr %0, i32 0, i32 0
  %1 = load <10 x float>, ptr %value, align 4
  %2 = load ptr, ptr %B.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix, ptr %2, i32 0, i32 0
  %3 = load <10 x float>, ptr %value1, align 4
  %4 = fadd <10 x float> %1, %3
  ret <10 x float> %4
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z22test_subtract_templatev() #0 {
entry:
  %Mat1 = alloca %struct.MyMatrix, align 4
  %Mat2 = alloca %struct.MyMatrix, align 4
  %call = call noundef <10 x float> @_Z8subtractIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(ptr noundef nonnull align 4 dereferenceable(40) %Mat1, ptr noundef nonnull align 4 dereferenceable(40) %Mat2)
  %value = getelementptr inbounds nuw %struct.MyMatrix, ptr %Mat1, i32 0, i32 0
  store <10 x float> %call, ptr %value, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef <10 x float> @_Z8subtractIfLj2ELj5EEN8MyMatrixIT_XT0_EXT1_EE8matrix_tERS2_S4_(ptr noundef nonnull align 4 dereferenceable(40) %A, ptr noundef nonnull align 4 dereferenceable(40) %B) #0 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  %0 = load ptr, ptr %A.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix, ptr %0, i32 0, i32 0
  %1 = load <10 x float>, ptr %value, align 4
  %2 = load ptr, ptr %B.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix, ptr %2, i32 0, i32 0
  %3 = load <10 x float>, ptr %value1, align 4
  %4 = fsub <10 x float> %1, %3
  ret <10 x float> %4
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z24test_DoubleWrapper1_Sub1R8MyMatrixIdLj10ELj9EE(ptr noundef nonnull align 8 dereferenceable(720) %m) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w1 = alloca %struct.DoubleWrapper1, align 4
  store ptr %m, ptr %m.addr, align 8
  %x = getelementptr inbounds nuw %struct.DoubleWrapper1, ptr %w1, i32 0, i32 0
  store i32 10, ptr %x, align 4
  %0 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %0, i32 0, i32 0
  %1 = load <90 x double>, ptr %value, align 8
  %call = call noundef double @_ZN14DoubleWrapper1cvdEv(ptr noundef nonnull align 4 dereferenceable(4) %w1)
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %call, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %2 = fsub <90 x double> %1, %scalar.splat.splat
  %3 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %3, i32 0, i32 0
  store <90 x double> %2, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef double @_ZN14DoubleWrapper1cvdEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #1 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %x = getelementptr inbounds nuw %struct.DoubleWrapper1, ptr %this1, i32 0, i32 0
  %0 = load i32, ptr %x, align 4
  %conv = sitofp i32 %0 to double
  ret double %conv
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z24test_DoubleWrapper1_Sub2R8MyMatrixIdLj10ELj9EE(ptr noundef nonnull align 8 dereferenceable(720) %m) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w1 = alloca %struct.DoubleWrapper1, align 4
  store ptr %m, ptr %m.addr, align 8
  %x = getelementptr inbounds nuw %struct.DoubleWrapper1, ptr %w1, i32 0, i32 0
  store i32 10, ptr %x, align 4
  %call = call noundef double @_ZN14DoubleWrapper1cvdEv(ptr noundef nonnull align 4 dereferenceable(4) %w1)
  %0 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %0, i32 0, i32 0
  %1 = load <90 x double>, ptr %value, align 8
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %call, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %2 = fsub <90 x double> %scalar.splat.splat, %1
  %3 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %3, i32 0, i32 0
  store <90 x double> %2, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z24test_DoubleWrapper2_Add1R8MyMatrixIdLj10ELj9EE(ptr noundef nonnull align 8 dereferenceable(720) %m) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w2 = alloca %struct.DoubleWrapper2, align 4
  store ptr %m, ptr %m.addr, align 8
  %x = getelementptr inbounds nuw %struct.DoubleWrapper2, ptr %w2, i32 0, i32 0
  store i32 20, ptr %x, align 4
  %0 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %0, i32 0, i32 0
  %1 = load <90 x double>, ptr %value, align 8
  %call = call noundef double @_ZN14DoubleWrapper2cvdEv(ptr noundef nonnull align 4 dereferenceable(4) %w2)
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %call, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %2 = fadd <90 x double> %1, %scalar.splat.splat
  %3 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %3, i32 0, i32 0
  store <90 x double> %2, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef double @_ZN14DoubleWrapper2cvdEv(ptr noundef nonnull align 4 dereferenceable(4) %this) #1 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %x = getelementptr inbounds nuw %struct.DoubleWrapper2, ptr %this1, i32 0, i32 0
  %0 = load i32, ptr %x, align 4
  %conv = sitofp i32 %0 to double
  ret double %conv
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z24test_DoubleWrapper2_Add2R8MyMatrixIdLj10ELj9EE(ptr noundef nonnull align 8 dereferenceable(720) %m) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w2 = alloca %struct.DoubleWrapper2, align 4
  store ptr %m, ptr %m.addr, align 8
  %x = getelementptr inbounds nuw %struct.DoubleWrapper2, ptr %w2, i32 0, i32 0
  store i32 20, ptr %x, align 4
  %call = call noundef double @_ZN14DoubleWrapper2cvdEv(ptr noundef nonnull align 4 dereferenceable(4) %w2)
  %0 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %0, i32 0, i32 0
  %1 = load <90 x double>, ptr %value, align 8
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %call, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %2 = fadd <90 x double> %scalar.splat.splat, %1
  %3 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %3, i32 0, i32 0
  store <90 x double> %2, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z19test_IntWrapper_AddR8MyMatrixIdLj10ELj9EE(ptr noundef nonnull align 8 dereferenceable(720) %m) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w3 = alloca %struct.IntWrapper, align 1
  store ptr %m, ptr %m.addr, align 8
  %x = getelementptr inbounds nuw %struct.IntWrapper, ptr %w3, i32 0, i32 0
  store i8 99, ptr %x, align 1
  %0 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %0, i32 0, i32 0
  %1 = load <90 x double>, ptr %value, align 8
  %call = call noundef i32 @_ZN10IntWrappercviEv(ptr noundef nonnull align 1 dereferenceable(1) %w3)
  %conv = sitofp i32 %call to double
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %conv, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %2 = fadd <90 x double> %1, %scalar.splat.splat
  %3 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %3, i32 0, i32 0
  store <90 x double> %2, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef i32 @_ZN10IntWrappercviEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %x = getelementptr inbounds nuw %struct.IntWrapper, ptr %this1, i32 0, i32 0
  %0 = load i8, ptr %x, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z19test_IntWrapper_SubR8MyMatrixIdLj10ELj9EE(ptr noundef nonnull align 8 dereferenceable(720) %m) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w3 = alloca %struct.IntWrapper, align 1
  store ptr %m, ptr %m.addr, align 8
  %x = getelementptr inbounds nuw %struct.IntWrapper, ptr %w3, i32 0, i32 0
  store i8 99, ptr %x, align 1
  %call = call noundef i32 @_ZN10IntWrappercviEv(ptr noundef nonnull align 1 dereferenceable(1) %w3)
  %conv = sitofp i32 %call to double
  %0 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %0, i32 0, i32 0
  %1 = load <90 x double>, ptr %value, align 8
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %conv, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %2 = fsub <90 x double> %scalar.splat.splat, %1
  %3 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %3, i32 0, i32 0
  store <90 x double> %2, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z22test_multiply_template8MyMatrixIfLj2ELj5EES_IfLj5ELj2EE(ptr dead_on_unwind noalias writable sret(%struct.MyMatrix.1) align 4 %agg.result, ptr noundef byval(%struct.MyMatrix) align 8 %Mat1, ptr noundef byval(%struct.MyMatrix.2) align 8 %Mat2) #2 {
entry:
  %call = call noundef <4 x float> @_Z8multiplyIfLj2ELj5ELj2EEN8MyMatrixIT_XT0_EXT2_EE8matrix_tERS0_IS1_XT0_EXT1_EERS0_IS1_XT1_EXT2_EE(ptr noundef nonnull align 4 dereferenceable(40) %Mat1, ptr noundef nonnull align 4 dereferenceable(40) %Mat2)
  %value = getelementptr inbounds nuw %struct.MyMatrix.1, ptr %agg.result, i32 0, i32 0
  store <4 x float> %call, ptr %value, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef <4 x float> @_Z8multiplyIfLj2ELj5ELj2EEN8MyMatrixIT_XT0_EXT2_EE8matrix_tERS0_IS1_XT0_EXT1_EERS0_IS1_XT1_EXT2_EE(ptr noundef nonnull align 4 dereferenceable(40) %A, ptr noundef nonnull align 4 dereferenceable(40) %B) #2 {
entry:
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  %0 = load ptr, ptr %A.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix, ptr %0, i32 0, i32 0
  %1 = load <10 x float>, ptr %value, align 4
  %2 = load ptr, ptr %B.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.2, ptr %2, i32 0, i32 0
  %3 = load <10 x float>, ptr %value1, align 4
  %4 = call <4 x float> @llvm.matrix.multiply.v4f32.v10f32.v10f32(<10 x float> %1, <10 x float> %3, i32 2, i32 5, i32 2)
  ret <4 x float> %4
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z24test_IntWrapper_MultiplyR8MyMatrixIdLj10ELj9EER10IntWrapper(ptr noundef nonnull align 8 dereferenceable(720) %m, ptr noundef nonnull align 1 dereferenceable(1) %w3) #1 {
entry:
  %m.addr = alloca ptr, align 8
  %w3.addr = alloca ptr, align 8
  store ptr %m, ptr %m.addr, align 8
  store ptr %w3, ptr %w3.addr, align 8
  %0 = load ptr, ptr %w3.addr, align 8
  %call = call noundef i32 @_ZN10IntWrappercviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
  %conv = sitofp i32 %call to double
  %1 = load ptr, ptr %m.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %1, i32 0, i32 0
  %2 = load <90 x double>, ptr %value, align 8
  %scalar.splat.splatinsert = insertelement <90 x double> poison, double %conv, i64 0
  %scalar.splat.splat = shufflevector <90 x double> %scalar.splat.splatinsert, <90 x double> poison, <90 x i32> zeroinitializer
  %3 = fmul <90 x double> %scalar.splat.splat, %2
  %4 = load ptr, ptr %m.addr, align 8
  %value1 = getelementptr inbounds nuw %struct.MyMatrix.0, ptr %4, i32 0, i32 0
  store <90 x double> %3, ptr %value1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z21test_insert_template1R8MyMatrixIjLj2ELj2EEjjj(ptr noundef nonnull align 4 dereferenceable(16) %Mat, i32 noundef %e, i32 noundef %i, i32 noundef %j) #1 {
entry:
  %Mat.addr = alloca ptr, align 8
  %e.addr = alloca i32, align 4
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  store ptr %Mat, ptr %Mat.addr, align 8
  store i32 %e, ptr %e.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  store i32 %j, ptr %j.addr, align 4
  %0 = load ptr, ptr %Mat.addr, align 8
  %1 = load i32, ptr %e.addr, align 4
  %2 = load i32, ptr %i.addr, align 4
  %3 = load i32, ptr %j.addr, align 4
  call void @_Z6insertIjLj2ELj2EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noundef nonnull align 4 dereferenceable(16) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_Z6insertIjLj2ELj2EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noundef nonnull align 4 dereferenceable(16) %Mat, i32 noundef %e, i32 noundef %i, i32 noundef %j) #1 {
entry:
  %Mat.addr = alloca ptr, align 8
  %e.addr = alloca i32, align 4
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  store ptr %Mat, ptr %Mat.addr, align 8
  store i32 %e, ptr %e.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  store i32 %j, ptr %j.addr, align 4
  %0 = load i32, ptr %e.addr, align 4
  %1 = load ptr, ptr %Mat.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.3, ptr %1, i32 0, i32 0
  %2 = load i32, ptr %i.addr, align 4
  %3 = zext i32 %2 to i64
  %4 = load i32, ptr %j.addr, align 4
  %5 = zext i32 %4 to i64
  %6 = mul i64 %5, 2
  %7 = add i64 %6, %3
  %8 = load <4 x i32>, ptr %value, align 4
  %matins = insertelement <4 x i32> %8, i32 %0, i64 %7
  store <4 x i32> %matins, ptr %value, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z21test_insert_template2R8MyMatrixIfLj3ELj8EEf(ptr noundef nonnull align 4 dereferenceable(96) %Mat, float noundef %e) #1 {
entry:
  %Mat.addr = alloca ptr, align 8
  %e.addr = alloca float, align 4
  store ptr %Mat, ptr %Mat.addr, align 8
  store float %e, ptr %e.addr, align 4
  %0 = load ptr, ptr %Mat.addr, align 8
  %1 = load float, ptr %e.addr, align 4
  call void @_Z6insertIfLj3ELj8EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noundef nonnull align 4 dereferenceable(96) %0, float noundef %1, i32 noundef 2, i32 noundef 5)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_Z6insertIfLj3ELj8EEvR8MyMatrixIT_XT0_EXT1_EES1_jj(ptr noundef nonnull align 4 dereferenceable(96) %Mat, float noundef %e, i32 noundef %i, i32 noundef %j) #1 {
entry:
  %Mat.addr = alloca ptr, align 8
  %e.addr = alloca float, align 4
  %i.addr = alloca i32, align 4
  %j.addr = alloca i32, align 4
  store ptr %Mat, ptr %Mat.addr, align 8
  store float %e, ptr %e.addr, align 4
  store i32 %i, ptr %i.addr, align 4
  store i32 %j, ptr %j.addr, align 4
  %0 = load float, ptr %e.addr, align 4
  %1 = load ptr, ptr %Mat.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.4, ptr %1, i32 0, i32 0
  %2 = load i32, ptr %i.addr, align 4
  %3 = zext i32 %2 to i64
  %4 = load i32, ptr %j.addr, align 4
  %5 = zext i32 %4 to i64
  %6 = mul i64 %5, 3
  %7 = add i64 %6, %3
  %8 = load <24 x float>, ptr %value, align 4
  %matins = insertelement <24 x float> %8, float %0, i64 %7
  store <24 x float> %matins, ptr %value, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define noundef i32 @_Z21test_extract_template8MyMatrixIiLj2ELj2EE(ptr noundef byval(%struct.MyMatrix.5) align 8 %Mat1) #1 {
entry:
  %call = call noundef i32 @_Z7extractIiLj2ELj2EET_R8MyMatrixIS0_XT0_EXT1_EE(ptr noundef nonnull align 4 dereferenceable(16) %Mat1)
  ret i32 %call
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef i32 @_Z7extractIiLj2ELj2EET_R8MyMatrixIS0_XT0_EXT1_EE(ptr noundef nonnull align 4 dereferenceable(16) %Mat) #1 {
entry:
  %Mat.addr = alloca ptr, align 8
  store ptr %Mat, ptr %Mat.addr, align 8
  %0 = load ptr, ptr %Mat.addr, align 8
  %value = getelementptr inbounds nuw %struct.MyMatrix.5, ptr %0, i32 0, i32 0
  %1 = load <4 x i32>, ptr %value, align 4
  %matrixext = extractelement <4 x i32> %1, i64 1
  ret i32 %matrixext
}

; Function Attrs: mustprogress noinline nounwind optnone
define noundef double @_Z21test_matrix_subscriptu11matrix_typeILm4ELm4EdE(<16 x double> noundef %m) #3 {
entry:
  %m.addr = alloca [16 x double], align 8
  store <16 x double> %m, ptr %m.addr, align 8
  %0 = load <16 x double>, ptr %m.addr, align 8
  %call = call noundef nonnull align 8 dereferenceable(8) ptr @_Z16matrix_subscriptIiiEDTixixfp_fp0_fp1_Eu11matrix_typeILm4ELm4EdET_T0_(<16 x double> noundef %0, i32 noundef 1, i32 noundef 2)
  %1 = load double, ptr %call, align 8
  ret double %1
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef nonnull align 8 dereferenceable(8) ptr @_Z16matrix_subscriptIiiEDTixixfp_fp0_fp1_Eu11matrix_typeILm4ELm4EdET_T0_(<16 x double> noundef %m, i32 noundef %r, i32 noundef %c) #3 {
entry:
  %m.addr = alloca [16 x double], align 8
  %r.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  store <16 x double> %m, ptr %m.addr, align 8
  store i32 %r, ptr %r.addr, align 4
  store i32 %c, ptr %c.addr, align 4
  ret ptr @_ZZ16matrix_subscriptIiiEDTixixfp_fp0_fp1_Eu11matrix_typeILm4ELm4EdET_T0_E1d
}

; Function Attrs: mustprogress noinline nounwind optnone
define noundef double @_Z22extract_IntWrapper_idxRu11matrix_typeILm4ELm4EdE10IntWrapper15UnsignedWrapper(ptr noundef nonnull align 8 dereferenceable(128) %m, i8 %i.coerce, i8 %j.coerce) #1 {
entry:
  %i = alloca %struct.IntWrapper, align 1
  %j = alloca %struct.UnsignedWrapper, align 1
  %m.addr = alloca ptr, align 8
  %coerce.dive = getelementptr inbounds nuw %struct.IntWrapper, ptr %i, i32 0, i32 0
  store i8 %i.coerce, ptr %coerce.dive, align 1
  %coerce.dive1 = getelementptr inbounds nuw %struct.UnsignedWrapper, ptr %j, i32 0, i32 0
  store i8 %j.coerce, ptr %coerce.dive1, align 1
  store ptr %m, ptr %m.addr, align 8
  %call = call noundef i32 @_ZN10IntWrappercviEv(ptr noundef nonnull align 1 dereferenceable(1) %i)
  %add = add nsw i32 %call, 1
  %0 = sext i32 %add to i64
  %call2 = call noundef i32 @_ZN15UnsignedWrappercvjEv(ptr noundef nonnull align 1 dereferenceable(1) %j)
  %sub = sub i32 %call2, 1
  %1 = zext i32 %sub to i64
  %2 = mul i64 %1, 4
  %3 = add i64 %2, %0
  %4 = load ptr, ptr %m.addr, align 8
  %5 = load <16 x double>, ptr %4, align 8
  %matrixext = extractelement <16 x double> %5, i64 %3
  ret double %matrixext
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef i32 @_ZN15UnsignedWrappercvjEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %x = getelementptr inbounds nuw %struct.UnsignedWrapper, ptr %this1, i32 0, i32 0
  %0 = load i8, ptr %x, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z15test_constexpr1Ru11matrix_typeILm4ELm4EfE(ptr noundef nonnull align 4 dereferenceable(64) %m) #4 {
entry:
  %m.addr = alloca ptr, align 8
  store ptr %m, ptr %m.addr, align 8
  %0 = load ptr, ptr %m.addr, align 8
  %1 = load <16 x float>, ptr %0, align 4
  %call = call noundef <16 x float> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIfLj4EEEv(ptr noundef nonnull align 1 dereferenceable(1) @_ZL11identmatrix)
  %2 = fadd <16 x float> %1, %call
  %3 = load ptr, ptr %m.addr, align 8
  store <16 x float> %2, ptr %3, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef <16 x float> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIfLj4EEEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #4 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  %result = alloca [16 x float], align 4
  %i = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp ne i32 %0, 4
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  %2 = zext i32 %1 to i64
  %3 = load i32, ptr %i, align 4
  %4 = zext i32 %3 to i64
  %5 = mul i64 %4, 4
  %6 = add i64 %5, %2
  %7 = load <16 x float>, ptr %result, align 4
  %matins = insertelement <16 x float> %7, float 1.000000e+00, i64 %6
  store <16 x float> %matins, ptr %result, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %8 = load i32, ptr %i, align 4
  %inc = add i32 %8, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !2

for.end:                                          ; preds = %for.cond
  %9 = load <16 x float>, ptr %result, align 4
  ret <16 x float> %9
}

; Function Attrs: mustprogress noinline nounwind optnone
define void @_Z15test_constexpr2Ru11matrix_typeILm5ELm5EiE(ptr noundef nonnull align 4 dereferenceable(100) %m) #5 {
entry:
  %m.addr = alloca ptr, align 8
  store ptr %m, ptr %m.addr, align 8
  %call = call noundef <25 x i32> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIiLj5EEEv(ptr noundef nonnull align 1 dereferenceable(1) @_ZL11identmatrix)
  %0 = load ptr, ptr %m.addr, align 8
  %1 = load <25 x i32>, ptr %0, align 4
  %2 = sub <25 x i32> %call, %1
  %3 = add <25 x i32> %2, splat (i32 1)
  %4 = load ptr, ptr %m.addr, align 8
  store <25 x i32> %3, ptr %4, align 4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr noundef <25 x i32> @_ZNK13identmatrix_tcvu11matrix_typeIXT0_EXT0_ET_EIiLj5EEEv(ptr noundef nonnull align 1 dereferenceable(1) %this) #5 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  %result = alloca [25 x i32], align 4
  %i = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp ne i32 %0, 5
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  %2 = zext i32 %1 to i64
  %3 = load i32, ptr %i, align 4
  %4 = zext i32 %3 to i64
  %5 = mul i64 %4, 5
  %6 = add i64 %5, %2
  %7 = load <25 x i32>, ptr %result, align 4
  %matins = insertelement <25 x i32> %7, i32 1, i64 %6
  store <25 x i32> %matins, ptr %result, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %8 = load i32, ptr %i, align 4
  %inc = add i32 %8, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !4

for.end:                                          ; preds = %for.cond
  %9 = load <25 x i32>, ptr %result, align 4
  ret <25 x i32> %9
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.matrix.multiply.v4f32.v10f32.v10f32(<10 x float>, <10 x float>, i32 immarg, i32 immarg, i32 immarg) #6

attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="320" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #2 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #3 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #4 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="512" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #5 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="800" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 20.1.8"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
!4 = distinct !{!4, !3}
