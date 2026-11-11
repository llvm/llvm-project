// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -disable-llvm-passes -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -disable-llvm-passes -o %t-cir.ll %s
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// A variadic declaration whose asm label names a definition of another type
// keeps its ellipsis when its calls are redirected to the definition.
extern int my_log(const char *fmt, ...) __asm__("real_log");

int test_plain(const char *p) { return my_log(p, 1, 2) + my_log(p); }

int real_log(const char *path, long x) { return (int)x; }

// CIR-LABEL: cir.func {{.*}} @test_plain(
// CIR:         %[[G1:.+]] = cir.get_global @real_log : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>>
// CIR-NEXT:    %[[P1:.+]] = cir.cast bitcast %[[G1]] : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>> -> !cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>
// CIR-NEXT:    %{{.+}} = cir.call %[[P1]](%{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>, !cir.ptr<!s8i>, !s32i, !s32i) -> !s32i
// CIR:         %[[G2:.+]] = cir.get_global @real_log : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>>
// CIR-NEXT:    %[[P2:.+]] = cir.cast bitcast %[[G2]] : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>> -> !cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>
// CIR-NEXT:    %{{.+}} = cir.call %[[P2]](%{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>, !cir.ptr<!s8i>) -> !s32i

// LLVM-LABEL:  define dso_local i32 @test_plain(
// LLVMCIR:       call i32 (ptr, ...) @real_log(ptr %{{[^,)]+}}, i32 1, i32 2)
// LLVMCIR:       call i32 (ptr, ...) @real_log(ptr %{{[^,)]+}})
// OGCG:          call i32 (ptr, ...) @real_log(ptr noundef %{{[^,)]+}}, i32 noundef 1, i32 noundef 2)
// OGCG:          call i32 (ptr, ...) @real_log(ptr noundef %{{[^,)]+}})

// The same holds when the definition is a gnu_inline body emitted later.
extern int xreal_vimpl(const char *path, long x);

extern __inline __attribute__((__always_inline__))
__attribute__((__gnu_inline__)) int
real_vimpl(const char *path, long x) {
  return xreal_vimpl(path, x);
}

extern int my_vprintf(const char *fmt, ...) __asm__("real_vimpl");

int test_inline(const char *p) {
  return my_vprintf(p, 1, 2) + my_vprintf(p);
}

// CIR-LABEL: cir.func {{.*}} @test_inline(
// CIR:         %[[G3:.+]] = cir.get_global @real_vimpl : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>>
// CIR-NEXT:    %[[P3:.+]] = cir.cast bitcast %[[G3]] : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>> -> !cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>
// CIR-NEXT:    %{{.+}} = cir.call %[[P3]](%{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>, !cir.ptr<!s8i>, !s32i, !s32i) -> !s32i
// CIR:         %[[G4:.+]] = cir.get_global @real_vimpl : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>>
// CIR-NEXT:    %[[P4:.+]] = cir.cast bitcast %[[G4]] : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !s64i) -> !s32i>> -> !cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>
// CIR-NEXT:    %{{.+}} = cir.call %[[P4]](%{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>, !cir.ptr<!s8i>) -> !s32i

// LLVM-LABEL:  define dso_local i32 @test_inline(
// LLVMCIR:       call i32 (ptr, ...) @real_vimpl(ptr %{{[^,)]+}}, i32 1, i32 2)
// LLVMCIR:       call i32 (ptr, ...) @real_vimpl(ptr %{{[^,)]+}})
// OGCG:          call i32 (ptr, ...) @real_vimpl(ptr noundef %{{[^,)]+}}, i32 noundef 1, i32 noundef 2)
// OGCG:          call i32 (ptr, ...) @real_vimpl(ptr noundef %{{[^,)]+}})

// A variadic declaration later defined as an alias is redirected the same way,
// and its calls keep the ellipsis.
extern int vlog(const char *fmt, ...);

int test_alias(const char *p) { return vlog(p, 1, 2) + vlog(p); }

int vlog_impl(const char *fmt, ...) { return 0; }
int vlog(const char *fmt, ...) __attribute__((alias("vlog_impl")));

// CIR-LABEL: cir.func {{.*}} @test_alias(
// CIR:         %[[G5:.+]] = cir.get_global @vlog : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>
// CIR-NEXT:    %{{.+}} = cir.call %[[G5]](%{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...) -> !s32i>>, !cir.ptr<!s8i>, !s32i, !s32i) -> !s32i
// CIR:         %{{.+}} = cir.call @vlog(%{{.+}}) : (!cir.ptr<!s8i>) -> !s32i

// LLVM-LABEL:  define dso_local i32 @test_alias(
// LLVMCIR:       call i32 (ptr, ...) @vlog(ptr %{{[^,)]+}}, i32 1, i32 2)
// LLVMCIR:       call i32 (ptr, ...) @vlog(ptr %{{[^,)]+}})
// OGCG:          call i32 (ptr, ...) @vlog(ptr noundef %{{[^,)]+}}, i32 noundef 1, i32 noundef 2)
// OGCG:          call i32 (ptr, ...) @vlog(ptr noundef %{{[^,)]+}})

// A library call CIRGen emits by name can bind to an unprototyped declaration.
// When a definition replaces that declaration, the call keeps its own operand
// types and stays non-variadic.
_Bool my_is_lock_free() __asm__("__atomic_is_lock_free");
_Bool use_noproto(void *p) { return my_is_lock_free(8UL, p); }
_Bool use_builtin(unsigned long n, void *p) {
  return __atomic_is_lock_free(n, p);
}
_Bool is_lock_free_def(unsigned long n, void *p) __asm__("__atomic_is_lock_free");
_Bool is_lock_free_def(unsigned long n, void *p) { return n <= 8; }

// CIR-LABEL: cir.func {{.*}} @use_builtin(
// CIR:         %[[G6:.+]] = cir.get_global @__atomic_is_lock_free : !cir.ptr<!cir.func<(!u64i, !cir.ptr<!void>) -> !cir.bool>>
// CIR-NEXT:    %{{.+}} = cir.call %[[G6]](%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!u64i, !cir.ptr<!void>) -> !cir.bool>>, !u64i, !cir.ptr<!void>) -> (!cir.bool {llvm.zeroext})

// LLVM-LABEL:  define dso_local zeroext i1 @use_builtin(
// LLVMCIR:       call zeroext i1 @__atomic_is_lock_free(i64 %{{[^,)]+}}, ptr %{{[^,)]+}})
// OGCG:          call zeroext i1 @__atomic_is_lock_free(i64 noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
