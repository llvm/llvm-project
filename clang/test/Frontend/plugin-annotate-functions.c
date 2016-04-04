// RUN: %clang -fplugin=%llvmshlibdir/AnnotateFunctions%pluginext -emit-llvm -DPRAGMA_ON -S %s -o - | FileCheck %s --check-prefix=PRAGMA
// RUN: %clang -fplugin=%llvmshlibdir/AnnotateFunctions%pluginext -emit-llvm -S %s -o - | FileCheck %s --check-prefix=NOPRAGMA
// RUN: not %clang -fplugin=%llvmshlibdir/AnnotateFunctions%pluginext -emit-llvm -DBAD_PRAGMA -S %s -o - 2>&1 | FileCheck %s --check-prefix=BADPRAGMA
// REQUIRES: plugins, examples

#ifdef PRAGMA_ON
#pragma enable_annotate
#endif

// BADPRAGMA: warning: extra tokens at end of #pragma directive
#ifdef BAD_PRAGMA
#pragma enable_annotate something
#endif

// PRAGMA: [[STR_VAR:@.+]] = private unnamed_addr constant [19 x i8] c"example_annotation\00"
// PRAGMA: @llvm.global.annotations = {{.*}}@fn1{{.*}}[[STR_VAR]]{{.*}}@fn2{{.*}}[[STR_VAR]]
// NOPRAGMA-NOT: [[STR_VAR:@.+]] = private unnamed_addr constant [19 x i8] c"example_annotation\00"
// NOPRAGMA-NOT: @llvm.global.annotations = {{.*}}@fn1{{.*}}[[STR_VAR]]{{.*}}@fn2{{.*}}[[STR_VAR]]
void fn1() { }
void fn2() { }

// BADPRAGMA: error: #pragma enable_annotate not allowed after declarations
#ifdef BAD_PRAGMA
#pragma enable_annotate
#endif
