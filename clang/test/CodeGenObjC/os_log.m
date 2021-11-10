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
extern __attribute__((visibility("default"))) NSString *GenString();
void os_log_pack_send(void *);

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log1(
// CHECK: alloca i8*, align 8
// CHECK: %[[A_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[OS_LOG_ARG:.*]] = alloca %{{.*}}*, align 8
// CHECK-O2: %[[V0:.*]] = call i8* @llvm.objc.retain(
// CHECK-O2: store i8* %[[V0]], i8** %[[A_ADDR]], align 8,
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[A_ADDR]], i8* %{{.*}})
// CHECK-O2: %[[V4:.*]] = call %{{.*}}* (...) @GenString() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL:.*]] = call %{{.*}}* (...) @GenString()
// CHECK-O0: %[[V2:.*]] = bitcast %{{.*}}* %[[CALL]] to i8*
// CHECK-O0: %[[V3:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V2]])
// CHECK-O0: %[[V4:.*]] = bitcast i8* %[[V3]] to %{{.*}}*
// CHECK: %[[V5:.*]] = bitcast %{{.*}}* %[[V4]] to i8*
// CHECK: %[[V6:.*]] = call i8* @llvm.objc.retain(i8* %[[V5]])
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to %{{.*}}*
// CHECK: store %{{.*}}* %[[V7]], %{{.*}}** %[[OS_LOG_ARG]],
// CHECK: %[[V8:.*]] = ptrtoint %{{.*}}* %[[V7]] to i64
// CHECK: %[[V9:.*]] = load i8*, i8** %[[A_ADDR]], align 8
// CHECK: %[[V10:.*]] = ptrtoint i8* %[[V9]] to i64
// CHECK: call void @__os_log_helper_1_2_2_8_64_8_64(i8* %{{.*}}, i64 %[[V8]], i64 %[[V10]])
// CHECK: %[[V11:.*]] = bitcast %{{.*}}* %[[V4]] to i8*
// CHECK: call void @llvm.objc.release(i8* %[[V11]])
// CHECK: call void @os_log_pack_send(i8* %{{.*}})
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(%{{.*}}* %[[V7]])
// CHECK-O2: %[[V13:.*]] = load %{{.*}}*, %{{.*}}** %[[OS_LOG_ARG]], align 8
// CHECK-O2: %[[V14:.*]] = bitcast %{{.*}}* %[[V13]] to i8*
// CHECK-O2: call void @llvm.objc.release(i8* %[[V14]])
// CHECK-O2: %[[V15:.*]] = load i8*, i8** %[[A_ADDR]], align 8
// CHECK-O2: call void @llvm.objc.release(i8* %[[V15]])
// CHECK-O0: %[[V12:.*]] = bitcast %{{.*}}** %[[OS_LOG_ARG]] to i8**
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[V12]], i8* null)
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[A_ADDR]], i8* null)

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
// CHECK: alloca i8*, align 8
// CHECK: %[[OS_LOG_ARG:.*]] = alloca i8*, align 8
// CHECK-O2: %[[V3:.*]] = call %{{.*}}* (...) @GenString() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL:.*]] = call %{{.*}}* (...) @GenString()
// CHECK-O0: %[[V1:.*]] = bitcast %{{.*}}* %[[CALL]] to i8*
// CHECK-O0: %[[V2:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V1]])
// CHECK-O0: %[[V3:.*]] = bitcast i8* %[[V2]] to %{{.*}}*
// CHECK: %[[V4:.*]] = bitcast %{{.*}}* %[[V3]] to i8*
// CHECK: %[[V5:.*]] = call i8* @llvm.objc.retain(i8* %[[V4]])
// CHECK: store i8* %[[V5]], i8** %[[OS_LOG_ARG]], align 8
// CHECK: %[[V6:.*]] = ptrtoint i8* %[[V5]] to i64
// CHECK: call void @__os_log_helper_1_2_1_8_64(i8* %{{.*}}, i64 %[[V6]])
// CHECK: %[[V7:.*]] = bitcast %{{.*}}* %[[V3]] to i8*
// CHECK: call void @llvm.objc.release(i8* %[[V7]])
// CHECK: call void @os_log_pack_send(i8* %{{.*}})
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(i8* %[[V5]])
// CHECK-O2: %[[V9:.*]] = load i8*, i8** %[[OS_LOG_ARG]], align 8
// CHECK-O2: call void @llvm.objc.release(i8* %[[V9]])
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[OS_LOG_ARG]], i8* null)

void test_builtin_os_log3(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@", (id)GenString());
  os_log_pack_send(buf);
}

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log4(
// CHECK: alloca i8*, align 8
// CHECK: %[[OS_LOG_ARG:.*]] = alloca i8*, align 8
// CHECK: %[[OS_LOG_ARG2:.*]] = alloca i8*, align 8
// CHECK-O2: %[[V4:.*]] = call {{.*}} @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL:.*]] = call {{.*}} @objc_msgSend
// CHECK-O0: %[[V4:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL]])
// CHECK: %[[V5:.*]] = call i8* @llvm.objc.retain(i8* %[[V4]])
// CHECK: store i8* %[[V5]], i8** %[[OS_LOG_ARG]], align 8
// CHECK: %[[V6:.*]] = ptrtoint i8* %[[V5]] to i64
// CHECK-O2: %[[V10:.*]] = call {{.*}} @objc_msgSend{{.*}} [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-O0: %[[CALL1:.*]] = call {{.*}} @objc_msgSend
// CHECK-O0: %[[V10:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL1]])
// CHECK: %[[V11:.*]] = call i8* @llvm.objc.retain(i8* %[[V10]])
// CHECK: store i8* %[[V11]], i8** %[[OS_LOG_ARG2]], align 8
// CHECK: %[[V12:.*]] = ptrtoint i8* %[[V11]] to i64
// CHECK: call void @__os_log_helper_1_2_2_8_64_8_64(i8* %{{.*}}, i64 %[[V6]], i64 %[[V12]])
// CHECK: call void @llvm.objc.release(i8* %[[V10]])
// CHECK: call void @llvm.objc.release(i8* %[[V4]])
// CHECK: call void @os_log_pack_send(i8* %{{.*}})
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(i8* %[[V11]])
// CHECK-O2: %[[V14:.*]] = load i8*, i8** %[[OS_LOG_ARG2]], align 8
// CHECK-O2: call void @llvm.objc.release(i8* %[[V14]])
// CHECK-O2: call void (...) @llvm.objc.clang.arc.use(i8* %[[V5]])
// CHECK-O2: %[[V15:.*]] = load i8*, i8** %[[OS_LOG_ARG]], align 8
// CHECK-O2: call void @llvm.objc.release(i8* %[[V15]])

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
