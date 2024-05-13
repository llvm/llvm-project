// RUN: %clang_cc1 -Werror -triple powerpc-unknown-unknown -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=PPC,PPC32
// RUN: %clang_cc1 -Werror -triple powerpc64le-unknown-linux -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=PPC,PPC64
// RUN: %clang_cc1 -Werror -triple powerpc64le-unknown-linux -emit-llvm -o - %s \
// RUN:   -target-cpu pwr8 | FileCheck %s --check-prefixes=PPC,PPC64
// RUN: %clang_cc1 -Werror -triple powerpc64-unknown-aix -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=PPC,AIX64
// RUN: %clang_cc1 -Werror -triple powerpc64-unknown-aix -emit-llvm -o - %s \
// RUN:   -target-cpu pwr8 | FileCheck %s --check-prefixes=PPC,AIX64

// PPC: @c = global i8 0, align 1{{$}}
_Atomic(char) c;

// PPC: @s = global i16 0, align 2{{$}}
_Atomic(short) s;

// PPC: @i = global i32 0, align 4{{$}}
_Atomic(int) i;

// PPC32: @l = global i32 0, align 4{{$}}
// PPC64: @l = global i64 0, align 8{{$}}
// AIX64: @l = global i64 0, align 8{{$}}
_Atomic(long) l;

// PPC: @ll = global i64 0, align 8{{$}}
_Atomic(long long) ll;

typedef struct {
  char x[8];
} O;

// PPC32: @o = global %struct.O zeroinitializer, align 1{{$}}
// PPC64: @o = global %struct.O zeroinitializer, align 8{{$}}
// AIX64: @o = global %struct.O zeroinitializer, align 8{{$}}
_Atomic(O) o;

typedef struct {
  char x[16];
} Q;

// PPC32: @q = global %struct.Q zeroinitializer, align 1{{$}}
// PPC64: @q = global %struct.Q zeroinitializer, align 16{{$}}
// AIX64: @q = global %struct.Q zeroinitializer, align 16{{$}}
_Atomic(Q) q;
