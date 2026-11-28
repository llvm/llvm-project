// RUN: %clang_cc1 -O2 -triple powerpc-ibm-aix -mloadtime-comment-vars=sccsid -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

#pragma comment(copyright, "@(#) pragma path")

static char *sccsid = "@(#) option path";

void f(void) {}

// CHECK: @[[PRAGMA:__loadtime_comment_str_[0-9a-f]+]] = weak_odr hidden unnamed_addr constant [17 x i8] c"@(#) pragma path\00", section "__loadtime_comment", align 1, !loadtime_comment ![[MD:[0-9]+]]
// CHECK: @sccsid = internal global ptr @.str, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK: @llvm.compiler.used = appending global [2 x ptr] [ptr @[[PRAGMA]], ptr @sccsid], section "llvm.metadata"
