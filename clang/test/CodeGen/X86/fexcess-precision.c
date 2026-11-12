// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast -target-feature +avx512fp16 \
// RUN: -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard -target-feature +avx512fp16 \
// RUN: -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast \
// RUN: -emit-llvm -ffp-eval-method=source -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=source -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard \
// RUN: -emit-llvm -ffp-eval-method=source -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard  -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=source -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -emit-llvm -ffp-eval-method=source -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=source -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast \
// RUN: -emit-llvm -ffp-eval-method=double -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=double -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard \
// RUN: -emit-llvm -ffp-eval-method=double -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=double -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -emit-llvm -ffp-eval-method=double -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=double -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast \
// RUN: -emit-llvm -ffp-eval-method=extended -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FP80 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=fast -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=extended -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FP80 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard \
// RUN: -emit-llvm -ffp-eval-method=extended -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FP80 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=standard -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=extended -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FP80 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -emit-llvm -ffp-eval-method=extended -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FP80 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -emit-llvm -ffp-eval-method=extended -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FP80 %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -ffp-contract=on -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -ffp-contract=on -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -fmath-errno -ffp-contract=on -fno-rounding-math \
// RUN: -ffp-eval-method=source -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -fmath-errno -ffp-contract=on -fno-rounding-math \
// RUN: -ffp-eval-method=source -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -fmath-errno -ffp-contract=on -fno-rounding-math \
// RUN: -ffp-eval-method=double -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -fmath-errno -ffp-contract=on -fno-rounding-math \
// RUN: -ffp-eval-method=double -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT-DBL %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -fmath-errno -ffp-contract=on -fno-rounding-math \
// RUN: -ffp-eval-method=extended -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -fmath-errno -ffp-contract=on -fno-rounding-math \
// RUN: -ffp-eval-method=extended -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-CONTRACT-EXT %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none \
// RUN: -fapprox-func -fmath-errno -fno-signed-zeros -mreassociate \
// RUN: -freciprocal-math -ffp-contract=on -fno-rounding-math \
// RUN: -funsafe-math-optimizations -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-UNSAFE %s

// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN: -ffloat16-excess-precision=none -target-feature +avx512fp16 \
// RUN: -fapprox-func -fmath-errno -fno-signed-zeros -mreassociate \
// RUN: -freciprocal-math -ffp-contract=on -fno-rounding-math \
// RUN: -funsafe-math-optimizations -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefixes=CHECK-UNSAFE %s

// CHECK-EXT-LABEL: @f(
// CHECK-EXT-NEXT:  entry:
// CHECK-EXT-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-EXT-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-EXT-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-EXT-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-EXT-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-EXT-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-EXT-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-EXT-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-EXT-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-EXT-NEXT:    [[EXT:%.*]] = fpext half [[TMP0]] to float
// CHECK-EXT-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-EXT-NEXT:    [[EXT1:%.*]] = fpext half [[TMP1]] to float
// CHECK-EXT-NEXT:    [[MUL:%.*]] = fmul float [[EXT]], [[EXT1]]
// CHECK-EXT-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-EXT-NEXT:    [[EXT2:%.*]] = fpext half [[TMP2]] to float
// CHECK-EXT-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-EXT-NEXT:    [[EXT3:%.*]] = fpext half [[TMP3]] to float
// CHECK-EXT-NEXT:    [[MUL4:%.*]] = fmul float [[EXT2]], [[EXT3]]
// CHECK-EXT-NEXT:    [[ADD:%.*]] = fadd float [[MUL]], [[MUL4]]
// CHECK-EXT-NEXT:    [[UNPROMOTION:%.*]] = fptrunc float [[ADD]] to half
// CHECK-EXT-NEXT:    ret half [[UNPROMOTION]]
//
// CHECK-NO-EXT-LABEL: @f(
// CHECK-NO-EXT-NEXT:  entry:
// CHECK-NO-EXT-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-NO-EXT-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-NO-EXT-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-NO-EXT-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-NO-EXT-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-NO-EXT-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-NO-EXT-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-NO-EXT-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-NO-EXT-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-NO-EXT-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-NO-EXT-NEXT:    [[MUL:%.*]] = fmul half [[TMP0]], [[TMP1]]
// CHECK-NO-EXT-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-NO-EXT-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-NO-EXT-NEXT:    [[MUL1:%.*]] = fmul half [[TMP2]], [[TMP3]]
// CHECK-NO-EXT-NEXT:    [[ADD:%.*]] = fadd half [[MUL]], [[MUL1]]
// CHECK-NO-EXT-NEXT:    ret half [[ADD]]
//
// CHECK-EXT-DBL-LABEL: @f(
// CHECK-EXT-DBL-NEXT:  entry:
// CHECK-EXT-DBL-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-EXT-DBL-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-EXT-DBL-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-EXT-DBL-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-EXT-DBL-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-EXT-DBL-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-EXT-DBL-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-EXT-DBL-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-EXT-DBL-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-EXT-DBL-NEXT:    [[CONV:%.*]] = fpext half [[TMP0]] to double
// CHECK-EXT-DBL-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-EXT-DBL-NEXT:    [[CONV1:%.*]] = fpext half [[TMP1]] to double
// CHECK-EXT-DBL-NEXT:    [[MUL:%.*]] = fmul double [[CONV]], [[CONV1]]
// CHECK-EXT-DBL-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-EXT-DBL-NEXT:    [[CONV2:%.*]] = fpext half [[TMP2]] to double
// CHECK-EXT-DBL-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-EXT-DBL-NEXT:    [[CONV3:%.*]] = fpext half [[TMP3]] to double
// CHECK-EXT-DBL-NEXT:    [[MUL4:%.*]] = fmul double [[CONV2]], [[CONV3]]
// CHECK-EXT-DBL-NEXT:    [[ADD:%.*]] = fadd double [[MUL]], [[MUL4]]
// CHECK-EXT-DBL-NEXT:    [[CONV5:%.*]] = fptrunc double [[ADD]] to half
// CHECK-EXT-DBL-NEXT:    ret half [[CONV5]]
//
// CHECK-EXT-FP80-LABEL: @f(
// CHECK-EXT-FP80-NEXT:  entry:
// CHECK-EXT-FP80-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-EXT-FP80-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-EXT-FP80-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-EXT-FP80-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-EXT-FP80-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-EXT-FP80-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-EXT-FP80-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-EXT-FP80-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-EXT-FP80-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-EXT-FP80-NEXT:    [[CONV:%.*]] = fpext half [[TMP0]] to x86_fp80
// CHECK-EXT-FP80-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-EXT-FP80-NEXT:    [[CONV1:%.*]] = fpext half [[TMP1]] to x86_fp80
// CHECK-EXT-FP80-NEXT:    [[MUL:%.*]] = fmul x86_fp80 [[CONV]], [[CONV1]]
// CHECK-EXT-FP80-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-EXT-FP80-NEXT:    [[CONV2:%.*]] = fpext half [[TMP2]] to x86_fp80
// CHECK-EXT-FP80-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-EXT-FP80-NEXT:    [[CONV3:%.*]] = fpext half [[TMP3]] to x86_fp80
// CHECK-EXT-FP80-NEXT:    [[MUL4:%.*]] = fmul x86_fp80 [[CONV2]], [[CONV3]]
// CHECK-EXT-FP80-NEXT:    [[ADD:%.*]] = fadd x86_fp80 [[MUL]], [[MUL4]]
// CHECK-EXT-FP80-NEXT:    [[CONV5:%.*]] = fptrunc x86_fp80 [[ADD]] to half
// CHECK-EXT-FP80-NEXT:    ret half [[CONV5]]
//
// CHECK-CONTRACT-LABEL: @f(
// CHECK-CONTRACT-NEXT:  entry:
// CHECK-CONTRACT-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-CONTRACT-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-CONTRACT-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-CONTRACT-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-CONTRACT-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-CONTRACT-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-CONTRACT-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-CONTRACT-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-CONTRACT-NEXT:    [[MUL1:%.*]] = fmul half [[TMP2]], [[TMP3]]
// CHECK-CONTRACT-NEXT:    [[TMP4:%.*]] = call half @llvm.fmuladd.f16(half [[TMP0]], half [[TMP1]], half [[MUL1]])
// CHECK-CONTRACT-NEXT:    ret half [[TMP4]]
//
// CHECK-CONTRACT-DBL-LABEL: @f(
// CHECK-CONTRACT-DBL-NEXT:  entry:
// CHECK-CONTRACT-DBL-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-DBL-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-DBL-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-DBL-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-DBL-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    [[CONV:%.*]] = fpext half [[TMP0]] to double
// CHECK-CONTRACT-DBL-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    [[CONV1:%.*]] = fpext half [[TMP1]] to double
// CHECK-CONTRACT-DBL-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    [[CONV2:%.*]] = fpext half [[TMP2]] to double
// CHECK-CONTRACT-DBL-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-CONTRACT-DBL-NEXT:    [[CONV3:%.*]] = fpext half [[TMP3]] to double
// CHECK-CONTRACT-DBL-NEXT:    [[MUL4:%.*]] = fmul double [[CONV2]], [[CONV3]]
// CHECK-CONTRACT-DBL-NEXT:    [[TMP4:%.*]] = call double @llvm.fmuladd.f64(double [[CONV]], double [[CONV1]], double [[MUL4]])
// CHECK-CONTRACT-DBL-NEXT:    [[CONV5:%.*]] = fptrunc double [[TMP4]] to half
// CHECK-CONTRACT-DBL-NEXT:    ret half [[CONV5]]
//
// CHECK-CONTRACT-EXT-LABEL: @f(
// CHECK-CONTRACT-EXT-NEXT:  entry:
// CHECK-CONTRACT-EXT-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-EXT-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-EXT-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-EXT-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-CONTRACT-EXT-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    [[CONV:%.*]] = fpext half [[TMP0]] to x86_fp80
// CHECK-CONTRACT-EXT-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    [[CONV1:%.*]] = fpext half [[TMP1]] to x86_fp80
// CHECK-CONTRACT-EXT-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    [[CONV2:%.*]] = fpext half [[TMP2]] to x86_fp80
// CHECK-CONTRACT-EXT-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-CONTRACT-EXT-NEXT:    [[CONV3:%.*]] = fpext half [[TMP3]] to x86_fp80
// CHECK-CONTRACT-EXT-NEXT:    [[MUL4:%.*]] = fmul x86_fp80 [[CONV2]], [[CONV3]]
// CHECK-CONTRACT-EXT-NEXT:    [[TMP4:%.*]] = call x86_fp80 @llvm.fmuladd.f80(x86_fp80 [[CONV]], x86_fp80 [[CONV1]], x86_fp80 [[MUL4]])
// CHECK-CONTRACT-EXT-NEXT:    [[CONV5:%.*]] = fptrunc x86_fp80 [[TMP4]] to half
// CHECK-CONTRACT-EXT-NEXT:    ret half [[CONV5]]
//
// CHECK-UNSAFE-LABEL: @f(
// CHECK-UNSAFE-NEXT:  entry:
// CHECK-UNSAFE-NEXT:    [[A_ADDR:%.*]] = alloca half
// CHECK-UNSAFE-NEXT:    [[B_ADDR:%.*]] = alloca half
// CHECK-UNSAFE-NEXT:    [[C_ADDR:%.*]] = alloca half
// CHECK-UNSAFE-NEXT:    [[D_ADDR:%.*]] = alloca half
// CHECK-UNSAFE-NEXT:    store half [[A:%.*]], ptr [[A_ADDR]]
// CHECK-UNSAFE-NEXT:    store half [[B:%.*]], ptr [[B_ADDR]]
// CHECK-UNSAFE-NEXT:    store half [[C:%.*]], ptr [[C_ADDR]]
// CHECK-UNSAFE-NEXT:    store half [[D:%.*]], ptr [[D_ADDR]]
// CHECK-UNSAFE-NEXT:    [[TMP0:%.*]] = load half, ptr [[A_ADDR]]
// CHECK-UNSAFE-NEXT:    [[TMP1:%.*]] = load half, ptr [[B_ADDR]]
// CHECK-UNSAFE-NEXT:    [[TMP2:%.*]] = load half, ptr [[C_ADDR]]
// CHECK-UNSAFE-NEXT:    [[TMP3:%.*]] = load half, ptr [[D_ADDR]]
// CHECK-UNSAFE-NEXT:    [[MUL1:%.*]] = fmul reassoc nsz arcp afn half [[TMP2]], [[TMP3]]
// CHECK-UNSAFE-NEXT:    [[TMP4:%.*]] = call reassoc nsz arcp afn half @llvm.fmuladd.f16(half [[TMP0]], half [[TMP1]], half [[MUL1]])
// CHECK-UNSAFE-NEXT:    ret half [[TMP4]]
//
_Float16 f(_Float16 a, _Float16 b, _Float16 c, _Float16 d) {
    return a * b + c * d;
}

// CHECK-EXT-LABEL: @getFEM(
// CHECK-EXT-NEXT:  entry:
// CHECK-EXT-NEXT:    ret i32 0
//
// CHECK-NO-EXT-LABEL: @getFEM(
// CHECK-NO-EXT-NEXT:  entry:
// CHECK-NO-EXT-NEXT:    ret i32 0
//
// CHECK-EXT-DBL-LABEL: @getFEM(
// CHECK-EXT-DBL-NEXT:  entry:
// CHECK-EXT-DBL-NEXT:    ret i32 1
//
// CHECK-EXT-FP80-LABEL: @getFEM(
// CHECK-EXT-FP80-NEXT:  entry:
// CHECK-EXT-FP80-NEXT:    ret i32 2
//
// CHECK-CONTRACT-LABEL: @getFEM(
// CHECK-CONTRACT-NEXT:  entry:
// CHECK-CONTRACT-NEXT:    ret i32 0
//
// CHECK-CONTRACT-DBL-LABEL: @getFEM(
// CHECK-CONTRACT-DBL-NEXT:  entry:
// CHECK-CONTRACT-DBL-NEXT:    ret i32 1
//
// CHECK-CONTRACT-EXT-LABEL: @getFEM(
// CHECK-CONTRACT-EXT-NEXT:  entry:
// CHECK-CONTRACT-EXT-NEXT:    ret i32 2
//
// CHECK-UNSAFE-LABEL: @getFEM(
// CHECK-UNSAFE-NEXT:  entry:
// CHECK-UNSAFE-NEXT:    ret i32 0
//
int getFEM() {
  return __FLT_EVAL_METHOD__;
}
