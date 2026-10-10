// Test -mloadtime-comment-vars= with a variable defined as an alias. Both the
// alias and the aliased variable are named in the list:
//   * the aliased variable owns the string, so it gets !loadtime_comment
//     metadata and is kept alive in llvm.compiler.used;
//   * the alias is diagnosed (covered by the Sema tests; silenced here with
//     -Wno-loadtime-comment-var) and is emitted as an ordinary alias, without
//     metadata.

// RUN: %clang_cc1 -O2 -triple powerpc-ibm-aix -Wno-loadtime-comment-var \
// RUN:   -mloadtime-comment-vars=sccsid,real -emit-llvm -disable-llvm-passes \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var \
// RUN:   -mloadtime-comment-vars=sccsid,real -emit-llvm -disable-llvm-passes \
// RUN:   -o - %s | FileCheck %s

static char real[] = "@(#) real string";
extern char sccsid[17] __attribute__((alias("real")));

void foo(void) {}

// CHECK: @real = internal global [17 x i8] c"@(#) real string\00", align 1, !loadtime_comment ![[MD:[0-9]+]]
// CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @real], section "llvm.metadata"
// CHECK: @sccsid = alias [17 x i8], ptr @real{{$}}
// CHECK: ![[MD]] = !{}
