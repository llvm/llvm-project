// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int a, b, c;
};

int get(int n, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, n);
  struct S s = __builtin_va_arg(args, struct S);
  __builtin_va_end(args);
  return s.a + s.b + s.c;
}

// Aggregate __builtin_va_arg expands into the x86-64 register-save-area dance.

// CIR-LABEL: cir.func {{.*}} @get(
// CIR:   cir.va_start
// CIR:   %{{.+}} = cir.scope {
// CIR:     %[[GP_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"}
// CIR:     %[[GP:.+]] = cir.load %[[GP_P]]
// CIR:     %[[LIMIT:.+]] = cir.const #cir.int<32> : !u32i
// CIR:     %[[FITS:.+]] = cir.cmp le %[[GP]], %[[LIMIT]]
// CIR:     cir.brcond %[[FITS]]
// CIR:     cir.get_member %{{.+}}[3] {name = "reg_save_area"}
// CIR:     cir.get_member %{{.+}}[2] {name = "overflow_arg_area"}
// CIR:     cir.yield %{{.+}} : !rec_S
// CIR:   } : !rec_S
// CIR-NOT: cir.va_arg

// LLVM-LABEL: define dso_local i32 @get(i32 noundef %{{.+}}, ...)
// LLVM:   call void @llvm.va_start
// LLVM:   %[[GP:.+]] = load i32, ptr
// LLVM:   icmp ule i32 %[[GP]], 32
// LLVM:   getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 3
// LLVM:   getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.+}}, i32 0, i32 2
// LLVM:   phi ptr
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr %{{.+}}, ptr %{{.+}}, i64 12, i1 false)
// LLVM-NOT: va_arg

// OGCG-LABEL: define dso_local i32 @get(i32 noundef %{{.+}}, ...)
// OGCG:   call void @llvm.va_start
// OGCG:   %[[GP:.+]] = load i32, ptr %{{.+}}, align 16
// OGCG:   icmp ule i32 %[[GP]], 32
// OGCG:   load ptr, ptr %{{.+}}, align 16
// OGCG:   load ptr, ptr %{{.+}}, align 8
// OGCG:   phi ptr
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.+}}, ptr align 4 %{{.+}}, i64 12, i1 false)
// OGCG-NOT: va_arg
