// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O2 -disable-llvm-passes | FileCheck %s --check-prefixes=CHECK,CHECK-O2
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O0 | FileCheck %s --check-prefixes=CHECK,CHECK-O0
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -O2 -disable-llvm-passes | FileCheck %s --check-prefix=CHECK-MRR

// Make sure we emit clang.arc.use before calling objc_release as part of the
// cleanup. This way we make sure the object will not be released until the
// end of the full expression.

// rdar://problem/24528966

@interface C
- (id)m0;
+ (id)m1;
@end

C *c;

@class NSString;
extern __attribute__((visibility("default"))) NSString *GenString(void);
void os_log_pack_send(void *);

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log1(
// CHECK: alloca ptr, align 8
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[OS_LOG_ARG:.*]] = alloca ptr, align 8
// CHECK-O2: %[[V0:.*]] = call ptr @llvm.objc.retain(
// CHECK-O2: store ptr %[[V0]], ptr %[[A_ADDR]], align 8,
// CHECK-O0: call void @llvm.objc.storeStrong(ptr %[[A_ADDR]], ptr %{{.*}})
// CHECK-O2: %[[V3:.*]] = call ptr @GenString() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL:.*]] = call ptr @GenString()
// CHECK-O0: %[[V3:.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL]])
// CHECK: %[[V6:.*]] = call ptr @llvm.objc.retain(ptr %[[V3]])
// CHECK: store ptr %[[V6]], ptr %[[OS_LOG_ARG]],
// CHECK: %[[V8:.*]] = ptrtoint ptr %[[V6]] to i64
// CHECK: %[[V9:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: %[[V10:.*]] = ptrtoint ptr %[[V9]] to i64
// CHECK: call void @__os_log_helper_1_2_2_8_64_8_64(ptr noundef %{{.*}}, i64 noundef %[[V8]], i64 noundef %[[V10]])
// CHECK: call void @llvm.objc.release(ptr %[[V3]])
// CHECK: call void @os_log_pack_send(ptr noundef %{{.*}})
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(ptr %[[V6]])
// CHECK-O2: %[[V13:.*]] = load ptr, ptr %[[OS_LOG_ARG]], align 8
// CHECK-O2: call void @llvm.objc.release(ptr %[[V13]])
// CHECK-O2: %[[V15:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK-O2: call void @llvm.objc.release(ptr %[[V15]])
// CHECK-O0: call void @llvm.objc.storeStrong(ptr %[[OS_LOG_ARG]], ptr null)
// CHECK-O0: call void @llvm.objc.storeStrong(ptr %[[A_ADDR]], ptr null)

// CHECK-MRR-LABEL: define{{.*}} void @test_builtin_os_log1(
// CHECK-MRR-NOT: call {{.*}} @llvm.objc
// CHECK-MRR: ret void

void test_builtin_os_log1(void *buf, id a) {
  __builtin_os_log_format(buf, "capabilities: %@ %@", GenString(), a);
  os_log_pack_send(buf);
}

// CHECK: define{{.*}} void @test_builtin_os_log2(
// CHECK-NOT: @llvm.objc.retain(

void test_builtin_os_log2(void *buf, id __unsafe_unretained a) {
  __builtin_os_log_format(buf, "capabilities: %@", a);
  os_log_pack_send(buf);
}

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log3(
// CHECK: alloca ptr, align 8
// CHECK: %[[OS_LOG_ARG:.*]] = alloca ptr, align 8
// CHECK-O2: %[[V2:.*]] = call ptr @GenString() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL:.*]] = call ptr @GenString()
// CHECK-O0: %[[V2:.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL]])
// CHECK: %[[V5:.*]] = call ptr @llvm.objc.retain(ptr %[[V2]])
// CHECK: store ptr %[[V5]], ptr %[[OS_LOG_ARG]], align 8
// CHECK: %[[V6:.*]] = ptrtoint ptr %[[V5]] to i64
// CHECK: call void @__os_log_helper_1_2_1_8_64(ptr noundef %{{.*}}, i64 noundef %[[V6]])
// CHECK: call void @llvm.objc.release(ptr %[[V2]])
// CHECK: call void @os_log_pack_send(ptr noundef %{{.*}})
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(ptr %[[V5]])
// CHECK-O2: %[[V9:.*]] = load ptr, ptr %[[OS_LOG_ARG]], align 8
// CHECK-O2: call void @llvm.objc.release(ptr %[[V9]])
// CHECK-O0: call void @llvm.objc.storeStrong(ptr %[[OS_LOG_ARG]], ptr null)

void test_builtin_os_log3(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@", (id)GenString());
  os_log_pack_send(buf);
}

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log4(
// CHECK: alloca ptr, align 8
// CHECK: %[[OS_LOG_ARG:.*]] = alloca ptr, align 8
// CHECK: %[[OS_LOG_ARG2:.*]] = alloca ptr, align 8
// CHECK-O2: %[[V4:.*]] = call {{.*}} @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL:.*]] = call {{.*}} @objc_msgSend
// CHECK-O0: %[[V4:.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL]])
// CHECK: %[[V5:.*]] = call ptr @llvm.objc.retain(ptr %[[V4]])
// CHECK: store ptr %[[V5]], ptr %[[OS_LOG_ARG]], align 8
// CHECK: %[[V6:.*]] = ptrtoint ptr %[[V5]] to i64
// CHECK-O2: %[[V10:.*]] = call {{.*}} @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL1:.*]] = call {{.*}} @objc_msgSend
// CHECK-O0: %[[V10:.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %[[CALL1]])
// CHECK: %[[V11:.*]] = call ptr @llvm.objc.retain(ptr %[[V10]])
// CHECK: store ptr %[[V11]], ptr %[[OS_LOG_ARG2]], align 8
// CHECK: %[[V12:.*]] = ptrtoint ptr %[[V11]] to i64
// CHECK: call void @__os_log_helper_1_2_2_8_64_8_64(ptr noundef %{{.*}}, i64 noundef %[[V6]], i64 noundef %[[V12]])
// CHECK: call void @llvm.objc.release(ptr %[[V10]])
// CHECK: call void @llvm.objc.release(ptr %[[V4]])
// CHECK: call void @os_log_pack_send(ptr noundef %{{.*}})
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(ptr %[[V11]])
// CHECK-O2: %[[V14:.*]] = load ptr, ptr %[[OS_LOG_ARG2]], align 8
// CHECK-O2: call void @llvm.objc.release(ptr %[[V14]])
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(ptr %[[V5]])
// CHECK-O2: %[[V15:.*]] = load ptr, ptr %[[OS_LOG_ARG]], align 8
// CHECK-O2: call void @llvm.objc.release(ptr %[[V15]])

void test_builtin_os_log4(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@ %@", [c m0], [C m1]);
  os_log_pack_send(buf);
}

// FIXME: Lifetime of GenString's return should be extended in this case too.
// CHECK-LABEL: define{{.*}} void @test_builtin_os_log5(
// CHECK: call void @os_log_pack_send(
// CHECK-NOT: call void @llvm.objc.release(

void test_builtin_os_log5(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@", (0, GenString()));
  os_log_pack_send(buf);
}
