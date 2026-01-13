// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -emit-cir -o %t-default.cir
// RUN: FileCheck --input-file=%t-default.cir %s -check-prefix=CIR-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -fno-common -emit-cir -o %t-no-common.cir
// RUN: FileCheck --input-file=%t-no-common.cir %s -check-prefix=CIR-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -fcommon -emit-cir -o %t-common.cir
// RUN: FileCheck --input-file=%t-common.cir %s -check-prefix=CIR-COMMON

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -emit-llvm -o %t-default-cir.ll
// RUN: FileCheck --input-file=%t-default-cir.ll %s -check-prefix=LLVM-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -fno-common -emit-llvm -o %t-no-common-cir.ll
// RUN: FileCheck --input-file=%t-no-common-cir.ll %s -check-prefix=LLVM-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -fcommon -emit-llvm -o %t-common-cir.ll
// RUN: FileCheck --input-file=%t-common-cir.ll %s -check-prefix=LLVM-COMMON

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o %t-default.ll
// RUN: FileCheck --input-file=%t-default.ll %s -check-prefix=OGCG-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fno-common -emit-llvm -o %t-no-common.ll
// RUN: FileCheck --input-file=%t-no-common.ll %s -check-prefix=OGCG-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fcommon -emit-llvm -o %t-common.ll
// RUN: FileCheck --input-file=%t-common.ll %s -check-prefix=OGCG-COMMON

const int a = 42;
// CIR-DEFAULT: cir.global constant external @a = #cir.int<42>
// LLVM-DEFAULT: @a = constant i32 42
// OGCG-DEFAULT: @a = constant i32 42

// CIR-COMMON: cir.global constant external @a
// LLVM-COMMON: @a = constant i32 42
// OGCG-COMMON: @a = constant i32 42

const int b __attribute__((common)) = 42;
// CIR-DEFAULT: cir.global constant external @b = #cir.int<42>
// LLVM-DEFAULT: @b = constant i32 42
// OGCG-DEFAULT: @b = constant i32 42

// CIR-COMMON: cir.global constant external @b = #cir.int<42>
// LLVM-COMMON: @b = constant i32 42
// OGCG-COMMON: @b = constant i32 42

const int c __attribute__((nocommon)) = 42;
// CIR-DEFAULT: cir.global constant external @c = #cir.int<42>
// LLVM-DEFAULT: @c = constant i32 42
// OGCG-DEFAULT: @c = constant i32 42

// CIR-COMMON: cir.global constant external @c = #cir.int<42>
// LLVM-COMMON: @c = constant i32 42
// OGCG-COMMON: @c = constant i32 42

int d = 11;
// CIR-DEFAULT: cir.global external @d = #cir.int<11>
// LLVM-DEFAULT: @d = global i32 11
// OGCG-DEFAULT: @d = global i32 11

// CIR-COMMON: cir.global external @d = #cir.int<11>
// LLVM-COMMON: @d = global i32 11
// OGCG-COMMON: @d = global i32 11

int e;
// CIR-DEFAULT: cir.global external @e = #cir.int<0>
// LLVM-DEFAULT: @e = global i32 0
// OGCG-DEFAULT: @e = global i32 0

// CIR-COMMON: cir.global common @e = #cir.int<0>
// LLVM-COMMON: @e = common global i32 0
// OGCG-COMMON: @e = common global i32 0


int f __attribute__((common));
// CIR-DEFAULT: cir.global common @f = #cir.int<0>
// LLVM-DEFAULT: @f = common global i32 0
// OGCG-DEFAULT: @f = common global i32 0

// CIR-COMMON: cir.global common @f
// LLVM-COMMON: @f = common global i32 0
// OGCG-COMMON: @f = common global i32 0

int g __attribute__((nocommon));
// CIR-DEFAULT: cir.global external @g = #cir.int<0>
// LLVM-DEFAULT: @g = global i32
// OGCG-DEFAULT: @g = global i32 0

// CIR-COMMON: cir.global external @g = #cir.int<0>
// LLVM-COMMON: @g = global i32 0
// OGCG-COMMON: @g = global i32 0

const int h;
// CIR-DEFAULT: cir.global constant external @h = #cir.int<0>
// LLVM-DEFAULT: @h = constant i32
// OGCG-DEFAULT: @h = constant i32 0

// CIR-COMMON: cir.global common @h = #cir.int<0>
// LLVM-COMMON: @h = common global i32 0
// OGCG-COMMON: @h = common global i32 0

typedef void* (*fn_t)(long a, long b, char *f, int c);
fn_t ABC __attribute__ ((nocommon));
// CIR-DEFAULT: cir.global external @ABC = #cir.ptr<null>
// LLVM-DEFAULT: @ABC = global ptr null
// OGCG-DEFAULT: @ABC = global ptr null

// CIR-COMMON: cir.global external @ABC = #cir.ptr<null>
// LLVM-COMMON: @ABC = global ptr null
// OGCG-COMMON: @ABC = global ptr null
