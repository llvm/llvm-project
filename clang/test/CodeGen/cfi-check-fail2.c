// __cfi_check_fail codegen when not all CFI checkers are enabled.
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -O0 -fsanitize-cfi-cross-dso \
// RUN:     -fsanitize=cfi-vcall \
// RUN:     -emit-llvm -o - %s | FileCheck %s

// Check that ignorelist does not affect generated code.
// RUN: echo "src:*" > %t-all.ignorelist
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux -O0 -fsanitize-cfi-cross-dso \
// RUN:     -fsanitize=cfi-vcall -fsanitize-ignorelist=%t-all.ignorelist \
// RUN:     -emit-llvm -o - %s | FileCheck %s

void caller(void (*f)(void)) {
  f();
}

// CHECK: define weak_odr hidden void @__cfi_check_fail(i8* noundef %0, i8* noundef %1)
// CHECK: store i8* %0, i8** %[[ALLOCA0:.*]], align 8
// CHECK: store i8* %1, i8** %[[ALLOCA1:.*]], align 8
// CHECK: %[[DATA:.*]] = load i8*, i8** %[[ALLOCA0]], align 8
// CHECK: %[[ADDR:.*]] = load i8*, i8** %[[ALLOCA1]], align 8
// CHECK: %[[ICMP_NOT_NULL:.*]] = icmp ne i8* %[[DATA]], null
// CHECK: br i1 %[[ICMP_NOT_NULL]], label %[[CONT0:.*]], label %[[TRAP:.*]],

// CHECK: [[TRAP]]:
// CHECK-NEXT:   call void @llvm.ubsantrap(i8 2)
// CHECK-NEXT:   unreachable

// CHECK: [[CONT0]]:
// CHECK:   %[[A:.*]] = bitcast i8* %[[DATA]] to { i8, { i8*, i32, i32 }, i8* }*
// CHECK:   %[[KINDPTR:.*]] = getelementptr {{.*}} %[[A]], i32 0, i32 0
// CHECK:   %[[KIND:.*]] = load i8, i8* %[[KINDPTR]], align 4
// CHECK:   %[[VTVALID0:.*]] = call i1 @llvm.type.test(i8* %[[ADDR]], metadata !"all-vtables")
// CHECK:   %[[VTVALID:.*]] = zext i1 %[[VTVALID0]] to i64
// CHECK:   %[[NOT_0:.*]] = icmp ne i8 %[[KIND]], 0
// CHECK:   br i1 %[[NOT_0]], label %[[CONT1:.*]], label %[[HANDLE0:.*]], !prof

// CHECK: [[HANDLE0]]:
// CHECK:   %[[DATA0:.*]] = ptrtoint i8* %[[DATA]] to i64,
// CHECK:   %[[ADDR0:.*]] = ptrtoint i8* %[[ADDR]] to i64,
// CHECK:   call void @__ubsan_handle_cfi_check_fail_abort(i64 %[[DATA0]], i64 %[[ADDR0]], i64 %[[VTVALID]])
// CHECK:   unreachable

// CHECK: [[CONT1]]:
// CHECK:   %[[NOT_1:.*]] = icmp ne i8 %[[KIND]], 1
// CHECK:   br i1 %[[NOT_1]], label %[[CONT2:.*]], label %[[HANDLE1:.*]], !nosanitize

// CHECK: [[HANDLE1]]:
// CHECK-NEXT:   call void @llvm.ubsantrap(i8 2)
// CHECK-NEXT:   unreachable

// CHECK: [[CONT2]]:
// CHECK:   %[[NOT_2:.*]] = icmp ne i8 %[[KIND]], 2
// CHECK:   br i1 %[[NOT_2]], label %[[CONT3:.*]], label %[[HANDLE2:.*]], !nosanitize

// CHECK: [[HANDLE2]]:
// CHECK-NEXT:   call void @llvm.ubsantrap(i8 2)
// CHECK-NEXT:   unreachable

// CHECK: [[CONT3]]:
// CHECK:   %[[NOT_3:.*]] = icmp ne i8 %[[KIND]], 3
// CHECK:   br i1 %[[NOT_3]], label %[[CONT4:.*]], label %[[HANDLE3:.*]], !nosanitize

// CHECK: [[HANDLE3]]:
// CHECK-NEXT:   call void @llvm.ubsantrap(i8 2)
// CHECK-NEXT:   unreachable

// CHECK: [[CONT4]]:
// CHECK:   %[[NOT_4:.*]] = icmp ne i8 %[[KIND]], 4
// CHECK:   br i1 %[[NOT_4]], label %[[CONT5:.*]], label %[[HANDLE4:.*]], !nosanitize

// CHECK: [[HANDLE4]]:
// CHECK-NEXT:   call void @llvm.ubsantrap(i8 2)
// CHECK-NEXT:   unreachable

// CHECK: [[CONT5]]:
// CHECK:   ret void
