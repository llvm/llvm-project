// Test that a variable named in -mloadtime-comment-vars= and defined in a
// precompiled header is still preserved when the PCH is used: Sema attaches
// the implicit attribute when the header is compiled, the attribute is
// serialized in the PCH, and the unreferenced definition must still reach
// CodeGen in the including compilation. The first invocation compiles the
// header textually as a baseline; the PCH pair must produce the same result.

// RUN: %clang_cc1 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid \
// RUN:   -include %s -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid \
// RUN:   -include-pch %t -emit-llvm -o - %s | FileCheck %s

#ifndef HEADER
#define HEADER

static char sccsid[] = "@(#) pch sccsid";

#else

void use(void) {}

#endif

// CHECK: @sccsid = internal global [16 x i8] c"@(#) pch sccsid\00", align 1, !loadtime_comment !0
// CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @sccsid], section "llvm.metadata"
