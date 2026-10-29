// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -disable-llvm-passes -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -disable-llvm-passes -o %t-cir.ll %s
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// This test mirrors what older glibc headers (e.g. GCC 8.5 / glibc 2.27 era)
// expand to with -D_FILE_OFFSET_BITS=64. There are two declarations sharing
// the same mangled symbol "real_impl":
//
//   1. `my_stat` is declared with `struct my_stat *` and asm-renamed to
//      "real_impl" - this is the GNU __REDIRECT_NTH pattern.
//   2. `real_impl` is declared with `struct my_stat64 *` and additionally
//      provided as an `extern __inline __gnu_inline__` definition with a
//      body - this is the FORTIFY/__extern_inline wrapper pattern.
//
// CIR materializes the FuncOp for "real_impl" lazily on first use. When
// `test` calls `my_stat(p, &s)`, the FuncOp gets created with the
// `(ptr, ptr<my_stat>) -> i32` signature. Later, `real_impl`'s inline body
// is materialized: this triggers `replaceUsesOfNonProtoTypeWithRealFunction`
// which retroactively rewires the existing call site onto the new FuncOp,
// whose signature is `(ptr, ptr<my_stat64>) -> i32`. Without special
// handling, the rewired direct call would carry mismatching operand types.
// The rewrite must instead fall back to an indirect call through a
// function-pointer bitcast, the same shape used at initial call emission
// time.

struct my_stat   { int  legacy_field; };
struct my_stat64 { long modern_field; };

extern int xreal_impl(const char *path, struct my_stat64 *buf);

extern __inline __attribute__((__always_inline__))
__attribute__((__gnu_inline__)) int
real_impl(const char *path, struct my_stat64 *buf) {
  return xreal_impl(path, buf);
}

extern int my_stat(const char *path, struct my_stat *buf) __asm__("real_impl");

int test(const char *p) {
  struct my_stat s;
  return my_stat(p, &s);
}

// CIR-LABEL: cir.func {{.*}} @test(
//
// After `real_impl`'s body is materialized, the FuncOp's signature is
// updated to `(ptr, ptr<my_stat64>) -> i32`. The pre-existing call site
// in `test` (which still has `(ptr, ptr<my_stat>)` operand types) gets
// rewritten to an indirect call through a function-pointer bitcast.
// CIR:         %[[GLOBAL:.+]] = cir.get_global @real_impl : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat64>) -> !s32i>>
// CIR-NEXT:    %[[PTR:.+]] = cir.cast bitcast %[[GLOBAL]] : !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat64>) -> !s32i>> -> !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat>) -> !s32i>>
// CIR-NEXT:    cir.call %[[PTR]](%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat>) -> !s32i>>, !cir.ptr<!s8i>, !cir.ptr<!rec_my_stat>) -> !s32i

// The inline definition of `real_impl` is emitted with `available_externally`
// linkage, taking `struct my_stat64 *`.
// CIR:       cir.func {{.*}} available_externally @real_impl(%{{.+}}: !cir.ptr<!s8i>{{.*}}, %{{.+}}: !cir.ptr<!rec_my_stat64>{{.*}}) -> !s32i

// LLVM-LABEL: define dso_local i32 @test(
// LLVM:         %{{.+}} = call i32 @real_impl(ptr {{.*}}%{{.+}}, ptr {{.*}}%{{.+}})
// LLVM:       define {{.*}}available_externally i32 @real_impl(

// OGCG-LABEL: define dso_local i32 @test(
// OGCG:         %{{.+}} = call i32 @real_impl(ptr {{.*}}%{{.+}}, ptr {{.*}}%{{.+}})
// OGCG:       define {{.*}}available_externally i32 @real_impl(
