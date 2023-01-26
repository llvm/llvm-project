// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'B *'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 35 }, {{.*}}* @[[CHAR]] }

struct A { int n; };
struct B { int n; };
struct C : A, B {};

void *f(C *c) {
  // CHECK:                             define {{.*}} i8* @{{.*}}(%struct.C* noundef %[[C:.*]]) {{.*}} {
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          %[[C_ADDR:.*]] = alloca %struct.C*
  // CHECK-NEXT:                          store %struct.C* %[[C]], %struct.C** %[[C_ADDR]]
  // CHECK-NEXT:                          %[[C_RELOAD:.*]] = load %struct.C*, %struct.C** %[[C_ADDR]]
  // CHECK-NEXT:                          %[[IS_NULL:.*]] = icmp eq %struct.C* %[[C_RELOAD]], null
  // CHECK-NEXT:                          br i1 %[[IS_NULL]], label %[[CAST_END:[^,]+]], label %[[CAST_NOT_NULL:[^,]+]]
  // CHECK:                             [[CAST_NOT_NULL]]:
  // CHECK-NOSANITIZE-NEXT:               %[[BITCAST:.*]] = bitcast %struct.C* %[[C_RELOAD]] to i8*
  // CHECK-NOSANITIZE-NEXT:               %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BITCAST]], i64 4
  // CHECK-NOSANITIZE-NEXT:               %[[BITCAST2:.*]] = bitcast i8* %[[ADD_PTR]] to %struct.B*
  // CHECK-NOSANITIZE-NEXT:               br label %[[CAST_END]]
  // CHECK-SANITIZE-NEXT:                 %[[PTRTOINT:.*]] = ptrtoint %struct.C* %[[C_RELOAD]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:                 %[[MASKEDPTR:.*]] = and i64 %[[PTRTOINT]], 3, !nosanitize
  // CHECK-SANITIZE-NEXT:                 %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:                 br i1 %[[MASKCOND]], label %[[CONT:[^,]+]], label %[[HANDLER_TYPE_MISMATCH:[^,]+]]
  // CHECK-SANITIZE:                    [[HANDLER_TYPE_MISMATCH]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:       call void @__ubsan_handle_type_mismatch_v1_abort(
  // CHECK-SANITIZE-RECOVER-NEXT:         call void @__ubsan_handle_type_mismatch_v1(
  // CHECK-SANITIZE-TRAP-NEXT:            call void @llvm.ubsantrap(
  // CHECK-SANITIZE-UNREACHABLE-NEXT:     unreachable, !nosanitize
  // CHECK-SANITIZE:                    [[CONT]]:
  // CHECK-SANITIZE-NEXT:                 %[[BITCAST:.*]] = bitcast %struct.C* %[[C_RELOAD]] to i8*
  // CHECK-SANITIZE-NEXT:                 %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BITCAST]], i64 4
  // CHECK-SANITIZE-NEXT:                 %[[BITCAST2:.*]] = bitcast i8* %[[ADD_PTR]] to %struct.B*
  // CHECK-SANITIZE-NEXT:                 br label %[[CAST_END]]
  // CHECK:                             [[CAST_END]]:
  // CHECK-NOSANITIZE-NEXT:               %[[CAST_RESULT:.*]] = phi %struct.B* [ %[[BITCAST2]], %[[CAST_NOT_NULL]] ], [ null, %[[ENTRY]] ]
  // CHECK-NOSANITIZE-NEXT:               %[[CAST_END_BITCAST:.*]] = bitcast %struct.B* %[[CAST_RESULT]] to i8*
  // CHECK-NOSANITIZE-NEXT:               call void @llvm.assume(i1 true) [ "align"(i8* %[[CAST_END_BITCAST]], i64 8) ]
  // CHECK-NOSANITIZE-NEXT:               ret i8* %[[CAST_END_BITCAST]]
  // CHECK-NOSANITIZE-NEXT:              }
  // CHECK-SANITIZE-NEXT:                 %[[CAST_RESULT:.*]] = phi %struct.B* [ %[[BITCAST2]], %[[CONT]] ], [ null, %[[ENTRY]] ]
  // CHECK-SANITIZE-NEXT:                 %[[CAST_END_BITCAST:.*]] = bitcast %struct.B* %[[CAST_RESULT]] to i8*
  // CHECK-SANITIZE-NEXT:                 %[[PTRINT:.*]] = ptrtoint i8* %[[CAST_END_BITCAST]] to i64
  // CHECK-SANITIZE-NEXT:                 %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 7
  // CHECK-SANITIZE-NEXT:                 %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:                 %[[PTRINT_DUP:.*]] = ptrtoint i8* %[[CAST_END_BITCAST]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:                 br i1 %[[MASKCOND]], label %[[CONT1:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                    [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 8, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:         call void @__ubsan_handle_alignment_assumption(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 8, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:            call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:     unreachable, !nosanitize
  // CHECK-SANITIZE:                    [[CONT1]]:
  // CHECK-SANITIZE-NEXT:                 call void @llvm.assume(i1 true) [ "align"(i8* %[[CAST_END_BITCAST]], i64 8) ]
  // CHECK-SANITIZE-NEXT:                 ret i8* %[[CAST_END_BITCAST]]
  // CHECK-SANITIZE-NEXT:                }
#line 100
  return __builtin_assume_aligned((B*)c, 8);
}
