// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=full -O0 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-O0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=thin -O0 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-O0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=full -O1 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=thin -O1 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=full -O2 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=thin -O2 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=full -O3 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=thin -O3 %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=full -Os %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=thin -Os %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=full -Oz %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-FULL-OPTIMIZED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o /dev/null -mllvm -verify-cfg-preserved=0 -fdebug-pass-manager -flto=thin -Oz %s 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-THIN-OPTIMIZED

// CHECK-FULL-O0: Running pass: AlwaysInlinerPass
// CHECK-FULL-O0-NEXT: Running analysis: InnerAnalysisManagerProxy
// CHECK-FULL-O0-NEXT: Running analysis: ProfileSummaryAnalysis
// CHECK-FULL-O0-NEXT: Running pass: CoroConditionalWrapper
// CHECK-FULL-O0-NEXT: Running pass: CanonicalizeAliasesPass
// CHECK-FULL-O0-NEXT: Running pass: NameAnonGlobalPass
// CHECK-FULL-O0-NEXT: Running pass: AnnotationRemarksPass
// CHECK-FULL-O0-NEXT: Running analysis: TargetLibraryAnalysis
// CHECK-FULL-O0-NEXT: Running pass: VerifierPass
// CHECK-FULL-O0-NEXT: Running analysis: VerifierAnalysis
// CHECK-FULL-O0-NEXT: Running pass: BitcodeWriterPass

// CHECK-THIN-O0: Running pass: AlwaysInlinerPass
// CHECK-THIN-O0-NEXT: Running analysis: InnerAnalysisManagerProxy
// CHECK-THIN-O0-NEXT: Running analysis: ProfileSummaryAnalysis
// CHECK-THIN-O0-NEXT: Running pass: CoroConditionalWrapper
// CHECK-THIN-O0-NEXT: Running pass: CanonicalizeAliasesPass
// CHECK-THIN-O0-NEXT: Running pass: NameAnonGlobalPass
// CHECK-THIN-O0-NEXT: Running pass: AnnotationRemarksPass
// CHECK-THIN-O0-NEXT: Running analysis: TargetLibraryAnalysis
// CHECK-THIN-O0-NEXT: Running pass: VerifierPass
// CHECK-THIN-O0-NEXT: Running analysis: VerifierAnalysis
// CHECK-THIN-O0-NEXT: Running pass: ThinLTOBitcodeWriterPass

// TODO: The LTO pre-link pipeline currently invokes
//       buildPerModuleDefaultPipeline(), which contains LoopVectorizePass.
//       This may change as the pipeline gets implemented.
// CHECK-FULL-OPTIMIZED: Running pass: LoopVectorizePass
// CHECK-FULL-OPTIMIZED: Running pass: BitcodeWriterPass

// The ThinLTO pre-link pipeline shouldn't contain passes like
// LoopVectorizePass.
// CHECK-THIN-OPTIMIZED-NOT: Running pass: LoopVectorizePass
// CHECK-THIN-OPTIMIZED: Running pass: CanonicalizeAliasesPass
// CHECK-THIN-OPTIMIZED: Running pass: NameAnonGlobalPass
// CHECK-THIN-OPTIMIZED: Running pass: ThinLTOBitcodeWriterPass

void Foo(void) {}
