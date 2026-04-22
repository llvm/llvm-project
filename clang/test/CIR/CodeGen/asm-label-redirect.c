// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -disable-llvm-passes -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -disable-llvm-passes -o %t-cir.ll %s
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// This test simulates the GNU __REDIRECT_NTH pattern from older glibc headers.

struct my_stat   { int  legacy_field; };
struct my_stat64 { long modern_field; };

extern int my_stat(const char *path, struct my_stat *buf) __asm__("real_impl");
extern int real_impl(const char *path, struct my_stat64 *buf);

int test(const char *p) {
  struct my_stat   s_old;
  struct my_stat64 s_new;
  int r1 = my_stat(p, &s_old);
  int r2 = real_impl(p, &s_new);
  return r1 + r2;
}

// Both declarations are mangled to the same symbol "real_impl". CIR
// materializes a single FuncOp whose signature is whichever declaration
// it sees first - here, the my_stat declaration.
//
// CIR-LABEL: cir.func private @real_impl(
// CIR-SAME:    !cir.ptr<!s8i> {{.*}},
// CIR-SAME:    !cir.ptr<!rec_my_stat> {{.*}}) -> !s32i

// CIR-LABEL: cir.func {{.*}} @test(
//
// The first call site uses `struct my_stat *`, which matches the FuncOp's
// stored signature, so it lowers to a direct call.
// CIR:         %[[R1:.*]] = cir.call @real_impl(%{{.+}}, %{{.+}}) :
// CIR-SAME:      (!cir.ptr<!s8i> {{.*}}, !cir.ptr<!rec_my_stat> {{.*}}) -> !s32i
//
// The second call site uses `struct my_stat64 *`. The FuncOp's stored
// signature does not match, so the function pointer is bitcast to the
// call site's expected signature and the call becomes indirect.
// CIR:         %[[GLOBAL:.*]] = cir.get_global @real_impl :
// CIR-SAME:      !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat>) -> !s32i>>
// CIR-NEXT:    %[[PTR:.*]] = cir.cast bitcast %[[GLOBAL]] :
// CIR-SAME:      !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat>) -> !s32i>>
// CIR-SAME:      -> !cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat64>) -> !s32i>>
// CIR-NEXT:    %[[R2:.*]] = cir.call %[[PTR]](%{{.+}}, %{{.+}}) :
// CIR-SAME:      (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, !cir.ptr<!rec_my_stat64>) -> !s32i>>,
// CIR-SAME:       !cir.ptr<!s8i> {{.*}}, !cir.ptr<!rec_my_stat64> {{.*}}) -> !s32i

// LLVM:        declare i32 @real_impl(ptr {{[^,]*}}, ptr {{.*}})
// LLVM-LABEL:  define dso_local i32 @test(
// LLVM:          %{{.+}} = call i32 @real_impl(ptr {{.*}}%{{.+}}, ptr {{.*}}%{{.+}})
// LLVM:          %{{.+}} = call i32 @real_impl(ptr {{.*}}%{{.+}}, ptr {{.*}}%{{.+}})

// OGCG-LABEL:  define dso_local i32 @test(
// OGCG:          %{{.+}} = call i32 @real_impl(ptr {{.*}}%{{.+}}, ptr {{.*}}%{{.+}})
// OGCG:          %{{.+}} = call i32 @real_impl(ptr {{.*}}%{{.+}}, ptr {{.*}}%{{.+}})
// OGCG:        declare i32 @real_impl(ptr {{[^,]*}}, ptr {{.*}})
