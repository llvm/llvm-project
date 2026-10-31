// Tests that __builtin_clz/__builtin_ctz respect isCLZForZeroUndef() per target.
// x86 returns true (default), AArch64 returns false.

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=AARCH64-CIR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=AARCH64-LLVM
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=AARCH64-LLVM

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=X86-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=X86-LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=X86-LLVM

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}

// AARCH64-CIR-LABEL: _Z16test_builtin_ctzj
// AARCH64-CIR:         cir.ctz %{{.+}} : !u32i
// AARCH64-CIR-NOT:     poison_zero

// AARCH64-LLVM-LABEL: _Z16test_builtin_ctzj
// AARCH64-LLVM:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 false)

// X86-CIR-LABEL: _Z16test_builtin_ctzj
// X86-CIR:         cir.ctz %{{.+}} poison_zero : !u32i

// X86-LLVM-LABEL: _Z16test_builtin_ctzj
// X86-LLVM:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}

// AARCH64-CIR-LABEL: _Z16test_builtin_clzj
// AARCH64-CIR:         cir.clz %{{.+}} : !u32i
// AARCH64-CIR-NOT:     poison_zero

// AARCH64-LLVM-LABEL: _Z16test_builtin_clzj
// AARCH64-LLVM:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 false)

// X86-CIR-LABEL: _Z16test_builtin_clzj
// X86-CIR:         cir.clz %{{.+}} poison_zero : !u32i

// X86-LLVM-LABEL: _Z16test_builtin_clzj
// X86-LLVM:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

int test_builtin_ctzg_fallback(unsigned x, int fb) {
  return __builtin_ctzg(x, fb);
}

// On both targets, the fallback case always uses poison_zero=true.

// AARCH64-CIR-LABEL: _Z26test_builtin_ctzg_fallbackji
// AARCH64-CIR:         %[[CTZ:.+]] = cir.ctz %{{.+}} poison_zero : !u32i
// AARCH64-CIR:         %[[ZERO:.+]] = cir.const #cir.int<0>
// AARCH64-CIR:         %[[ISZERO:.+]] = cir.cmp eq %{{.+}}, %[[ZERO]] : !u32i
// AARCH64-CIR:         cir.select if %[[ISZERO]]

// AARCH64-LLVM-LABEL: _Z26test_builtin_ctzg_fallbackji
// AARCH64-LLVM:         %[[CTZ:.+]] = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)
// AARCH64-LLVM:         %[[ISZERO:.+]] = icmp eq i32 %{{.+}}, 0
// AARCH64-LLVM:         select i1 %[[ISZERO]], i32 %{{.+}}, i32 %[[CTZ]]

// X86-CIR-LABEL: _Z26test_builtin_ctzg_fallbackji
// X86-CIR:         %[[CTZ:.+]] = cir.ctz %{{.+}} poison_zero : !u32i
// X86-CIR:         %[[ZERO:.+]] = cir.const #cir.int<0>
// X86-CIR:         %[[ISZERO:.+]] = cir.cmp eq %{{.+}}, %[[ZERO]] : !u32i
// X86-CIR:         cir.select if %[[ISZERO]]

// X86-LLVM-LABEL: _Z26test_builtin_ctzg_fallbackji
// X86-LLVM:         %[[CTZ:.+]] = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)
// X86-LLVM:         %[[ISZERO:.+]] = icmp eq i32 %{{.+}}, 0
// X86-LLVM:         select i1 %[[ISZERO]], i32 %{{.+}}, i32 %[[CTZ]]

int test_builtin_clzg_fallback(unsigned x, int fb) {
  return __builtin_clzg(x, fb);
}

// AARCH64-CIR-LABEL: _Z26test_builtin_clzg_fallbackji
// AARCH64-CIR:         %[[CLZ:.+]] = cir.clz %{{.+}} poison_zero : !u32i
// AARCH64-CIR:         %[[ZERO:.+]] = cir.const #cir.int<0>
// AARCH64-CIR:         %[[ISZERO:.+]] = cir.cmp eq %{{.+}}, %[[ZERO]] : !u32i
// AARCH64-CIR:         cir.select if %[[ISZERO]]

// AARCH64-LLVM-LABEL: _Z26test_builtin_clzg_fallbackji
// AARCH64-LLVM:         %[[CLZ:.+]] = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)
// AARCH64-LLVM:         %[[ISZERO:.+]] = icmp eq i32 %{{.+}}, 0
// AARCH64-LLVM:         select i1 %[[ISZERO]], i32 %{{.+}}, i32 %[[CLZ]]

// X86-CIR-LABEL: _Z26test_builtin_clzg_fallbackji
// X86-CIR:         %[[CLZ:.+]] = cir.clz %{{.+}} poison_zero : !u32i
// X86-CIR:         %[[ZERO:.+]] = cir.const #cir.int<0>
// X86-CIR:         %[[ISZERO:.+]] = cir.cmp eq %{{.+}}, %[[ZERO]] : !u32i
// X86-CIR:         cir.select if %[[ISZERO]]

// X86-LLVM-LABEL: _Z26test_builtin_clzg_fallbackji
// X86-LLVM:         %[[CLZ:.+]] = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)
// X86-LLVM:         %[[ISZERO:.+]] = icmp eq i32 %{{.+}}, 0
// X86-LLVM:         select i1 %[[ISZERO]], i32 %{{.+}}, i32 %[[CLZ]]
