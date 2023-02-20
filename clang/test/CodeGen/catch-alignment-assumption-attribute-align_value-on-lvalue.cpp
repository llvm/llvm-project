// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

typedef char **__attribute__((align_value(0x100000000))) aligned_char;

struct ac_struct {
  // CHECK:  %[[STRUCT_AC_STRUCT:.*]] = type { ptr }
  aligned_char a;
};

// CHECK-SANITIZE-ANYRECOVER: @[[ALIGNED_CHAR:.*]] = {{.*}} c"'aligned_char' (aka 'char **')\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 13 }, {{.*}} @[[ALIGNED_CHAR]] }

char **load_from_ac_struct(struct ac_struct *x) {
  // CHECK:                           define{{.*}} ptr @{{.*}}(ptr noundef %[[X:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[STRUCT_AC_STRUCT_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                        store ptr %[[X]], ptr %[[STRUCT_AC_STRUCT_ADDR]], align 8
  // CHECK-NEXT:                        %[[X_RELOADED:.*]] = load ptr, ptr %[[STRUCT_AC_STRUCT_ADDR]], align 8
  // CHECK:                             %[[A_ADDR:.*]] = getelementptr inbounds %[[STRUCT_AC_STRUCT]], ptr %[[X_RELOADED]], i32 0, i32 0
  // CHECK:                             %[[A:.*]] = load ptr, ptr %[[A_ADDR]], align 8
  // CHECK-SANITIZE-NEXT:               %[[PTRINT:.*]] = ptrtoint ptr %[[A]] to i64
  // CHECK-SANITIZE-NEXT:               %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 4294967295
  // CHECK-SANITIZE-NEXT:               %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint ptr %[[A]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(ptr @[[LINE_100_ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 4294967296, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(ptr @[[LINE_100_ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 4294967296, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        call void @llvm.assume(i1 true) [ "align"(ptr %[[A]], i64 4294967296) ]
  // CHECK-NEXT:                        ret ptr %[[A]]
  // CHECK-NEXT:                      }
#line 100
  return x->a;
}
