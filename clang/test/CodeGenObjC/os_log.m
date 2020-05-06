// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O2 -fno-experimental-new-pass-manager | FileCheck %s --check-prefixes=CHECK,CHECK-LEGACY
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O2 -fexperimental-new-pass-manager | FileCheck %s --check-prefixes=CHECK,CHECK-NEWPM
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O0 | FileCheck %s -check-prefix=CHECK-O0
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -O2 -disable-llvm-passes | FileCheck %s -check-prefix=CHECK-MRR

// Make sure we emit clang.arc.use before calling objc_release as part of the
// cleanup. This way we make sure the object will not be released until the
// end of the full expression.

// rdar://problem/24528966

@interface C
- (id)m0;
+ (id)m1;
@end

@class NSString;
extern __attribute__((visibility("default"))) NSString *GenString();

// Behavior of __builtin_os_log differs between platforms, so only test on X86
#ifdef __x86_64__
// CHECK-LABEL: define i8* @test_builtin_os_log
// CHECK-O0-LABEL: define i8* @test_builtin_os_log
// CHECK: (i8* returned %[[BUF:.*]])
// CHECK-O0: (i8* %[[BUF:.*]])
void *test_builtin_os_log(void *buf) {
  return __builtin_os_log_format(buf, "capabilities: %@", GenString());

  // CHECK: %[[CALL:.*]] = tail call %[[TY0:.*]]* (...) @GenString()
  // CHECK: %[[V0:.*]] = bitcast %[[TY0]]* %[[CALL]] to i8*
  // CHECK: %[[V1:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V0]])
  // CHECK-LEGACY: %[[V2:.*]] = ptrtoint %[[TY0]]* %[[CALL]] to i64
  // CHECK-NEWPM: %[[RETAINED:.*]] = tail call i8* @llvm.objc.retain(i8* %[[V1]])
  // CHECK-NEWPM: %[[V2:.*]] = ptrtoint i8* %[[RETAINED]] to i64
  // CHECK: store i8 2, i8* %[[BUF]], align 1
  // CHECK: %[[NUMARGS_I:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
  // CHECK: store i8 1, i8* %[[NUMARGS_I]], align 1
  // CHECK: %[[ARGDESCRIPTOR_I:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
  // CHECK: store i8 64, i8* %[[ARGDESCRIPTOR_I]], align 1
  // CHECK: %[[ARGSIZE_I:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
  // CHECK: store i8 8, i8* %[[ARGSIZE_I]], align 1
  // CHECK: %[[ARGDATA_I:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
  // CHECK: %[[ARGDATACAST_I:.*]] = bitcast i8* %[[ARGDATA_I]] to i64*
  // CHECK: store i64 %[[V2]], i64* %[[ARGDATACAST_I]], align 1
  // CHECK-LEGACY: tail call void (...) @llvm.objc.clang.arc.use(%[[TY0]]* %[[CALL]])
  // CHECK-LEGACY: tail call void @llvm.objc.release(i8* %[[V0]])
  // CHECK-NEWPM: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[RETAINED]])
  // CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[RETAINED]])
  // CHECK: ret i8* %[[BUF]]

  // clang.arc.use is used and removed in IR optimizations. At O0, we should not
  // emit clang.arc.use, since it will not be removed and we will have a link
  // error.
  // CHECK-O0: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK-O0: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK-O0: %[[V0:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK-O0: %[[CALL:.*]] = call %[[TY0:.*]]* (...) @GenString()
  // CHECK-O0: %[[V1:.*]] = bitcast %[[TY0]]* %[[CALL]] to i8*
  // CHECK-O0: %[[V2:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V1]])
  // CHECK-O0: %[[V3:.*]] = bitcast i8* %[[V2]] to %[[TY0]]*
  // CHECK-O0: %[[V4:.*]] = bitcast %{{.*}}* %[[V3]] to i8*
  // CHECK-O0: %[[V5:.*]] = call i8* @llvm.objc.retain(i8* %[[V4]])
  // CHECK-O0: %[[V6:.*]] = bitcast i8* %[[V5]] to %{{.*}}*
  // CHECK-O0: %[[V7:.*]] = ptrtoint %{{.*}}* %[[V6]] to i64
  // CHECK-O0: call void @__os_log_helper_1_2_1_8_64(i8* %[[V0]], i64 %[[V7]])
  // CHECK-O0: %[[V5:.*]] = bitcast %[[TY0]]* %[[V3]] to i8*
  // CHECK-O0-NOT: call void (...) @llvm.objc.clang.arc.use({{.*}}
  // CHECK-O0: call void @llvm.objc.release(i8* %[[V5]])
  // CHECK-O0: ret i8* %[[V0]]
}

// CHECK-O0-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_8_64
// CHECK-O0: (i8* %[[BUFFER:.*]], i64 %[[ARG0:.*]])

// CHECK-O0: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK-O0: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK-O0: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK-O0: store i64 %[[ARG0]], i64* %[[ARG0_ADDR]], align 8
// CHECK-O0: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK-O0: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK-O0: store i8 2, i8* %[[SUMMARY]], align 1
// CHECK-O0: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK-O0: store i8 1, i8* %[[NUMARGS]], align 1
// CHECK-O0: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK-O0: store i8 64, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK-O0: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK-O0: store i8 8, i8* %[[ARGSIZE]], align 1
// CHECK-O0: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK-O0: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i64*
// CHECK-O0: %[[V0:.*]] = load i64, i64* %[[ARG0_ADDR]], align 8
// CHECK-O0: store i64 %[[V0]], i64* %[[ARGDATACAST]], align 1

void os_log_pack_send(void *);

// CHECK-NEWPM: define void @test_builtin_os_log2(i8* %[[BUF:.*]], i8* %[[A:.*]])
// CHECK-NEWPM: %[[V0:.*]] = tail call i8* @llvm.objc.retain(i8* %[[A]])
// CHECK-NEWPM: %[[CALL:.*]] = tail call %{{.*}}* (...) @GenString()
// CHECK-NEWPM: %[[V1:.*]] = bitcast %{{.*}}* %[[CALL]] to i8*
// CHECK-NEWPM: %[[V2:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V1]])
// CHECK-NEWPM: %[[V3:.*]] = tail call i8* @llvm.objc.retain(i8* %[[V2]])
// CHECK-NEWPM: %[[V4:.*]] = ptrtoint i8* %[[V3]] to i64
// CHECK-NEWPM: %[[V6:.*]] = ptrtoint i8* %[[V0]] to i64
// CHECK-NEWPM: %[[ARGDATA_I:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK-NEWPM: %[[ARGDATACAST_I:.*]] = bitcast i8* %[[ARGDATA_I]] to i64*
// CHECK-NEWPM: store i64 %[[V4]], i64* %[[ARGDATACAST_I]], align 1
// CHECK-NEWPM: %[[ARGDATA3_I:.*]] = getelementptr i8, i8* %[[BUF]], i64 14
// CHECK-NEWPM: %[[ARGDATACAST4_I:.*]] = bitcast i8* %[[ARGDATA3_I]] to i64*
// CHECK-NEWPM: store i64 %[[V6]], i64* %[[ARGDATACAST4_I]], align 1
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V2]])
// CHECK-NEWPM: tail call void @os_log_pack_send(i8* nonnull %[[BUF]])
// CHECK-NEWPM: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[V3]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V3]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V0]])

// CHECK-O0: define void @test_builtin_os_log2(i8* %{{.*}}, i8* %[[A:.*]])
// CHECK-O0: alloca i8*, align 8
// CHECK-O0: %[[A_ADDR:.*]] = alloca i8*, align 8
// CHECK-O0: %[[OS_LOG_ARG:.*]] = alloca %{{.*}}*, align 8
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[A_ADDR]], i8* %[[A]])
// CHECK-O0: %[[CALL:.*]] = call %{{.*}}* (...) @GenString()
// CHECK-O0: %[[V1:.*]] = bitcast %{{.*}}* %[[CALL]] to i8*
// CHECK-O0: %[[V2:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V1]]) #2
// CHECK-O0: %[[V3:.*]] = bitcast i8* %[[V2]] to %{{.*}}*
// CHECK-O0: %[[V4:.*]] = bitcast %{{.*}}* %[[V3]] to i8*
// CHECK-O0: %[[V5:.*]] = call i8* @llvm.objc.retain(i8* %[[V4]]) #2
// CHECK-O0: %[[V6:.*]] = bitcast i8* %[[V5]] to %{{.*}}*
// CHECK-O0: store %{{.*}}* %[[V6]], %{{.*}}** %[[OS_LOG_ARG]], align 8
// CHECK-O0: %[[V7:.*]] = ptrtoint %{{.*}}* %[[V6]] to i64
// CHECK-O0: %[[V8:.*]] = load i8*, i8** %[[A_ADDR]], align 8
// CHECK-O0: %[[V10:.*]] = ptrtoint i8* %[[V8]] to i64
// CHECK-O0: call void @__os_log_helper_1_2_2_8_64_8_64(i8* %{{.*}}, i64 %[[V7]], i64 %[[V10]])
// CHECK-O0: %[[V11:.*]] = bitcast %{{.*}}* %[[V3]] to i8*
// CHECK-O0: call void @llvm.objc.release(i8* %[[V11]])
// CHECK-O0: call void @os_log_pack_send(
// CHECK-O0: %[[V13:.*]] = bitcast %{{.*}}** %[[OS_LOG_ARG]] to i8**
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[V13]], i8* null)
// CHECK-O0: call void @llvm.objc.storeStrong(i8** %[[A_ADDR]], i8* null)

// CHECK-MRR: define void @test_builtin_os_log2(
// CHECK-MRR-NOT: call {{.*}} @llvm.objc

void test_builtin_os_log2(void *buf, id a) {
  __builtin_os_log_format(buf, "capabilities: %@ %@", GenString(), a);
  os_log_pack_send(buf);
}

// CHECK-O0: define void @test_builtin_os_log3(
// CHECK-O0-NOT: @llvm.objc.retain(

void test_builtin_os_log3(void *buf, id __unsafe_unretained a) {
  __builtin_os_log_format(buf, "capabilities: %@", a);
  os_log_pack_send(buf);
}

// CHECK-NEWPM: define void @test_builtin_os_log4(
// CHECK-NEWPM: %[[CALL:.*]] = tail call %{{.*}}* (...) @GenString()
// CHECK-NEWPM: %[[V0:.*]] = bitcast %{{.*}}* %[[CALL]] to i8*
// CHECK-NEWPM: %[[V1:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V0]])
// CHECK-NEWPM: %[[V2:.*]] = tail call i8* @llvm.objc.retain(i8* %[[V1]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V1]])
// CHECK-NEWPM: tail call void @os_log_pack_send(
// CHECK-NEWPM: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[V2]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V2]])

// CHECK-O0: define void @test_builtin_os_log4(

void test_builtin_os_log4(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@", (id)GenString());
  os_log_pack_send(buf);
}

C *c;

// CHECK-NEWPM: define void @test_builtin_os_log5(
// CHECK-NEWPM: %[[CALL:.*]] = tail call {{.*}} @objc_msgSend
// CHECK-NEWPM: %[[V3:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL]])
// CHECK-NEWPM: %[[V4:.*]] = tail call i8* @llvm.objc.retain(i8* %[[V3]])
// CHECK-NEWPM: %[[CALL1:.*]] = tail call {{.*}} @objc_msgSend
// CHECK-NEWPM: %[[V9:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[CALL1]])
// CHECK-NEWPM: %[[V10:.*]] = tail call i8* @llvm.objc.retain(i8* %[[V9]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V9]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V3]])
// CHECK-NEWPM: tail call void @os_log_pack_send(
// CHECK-NEWPM: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[V10]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V10]])
// CHECK-NEWPM: tail call void (...) @llvm.objc.clang.arc.use(i8* %[[V4]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V4]])

void test_builtin_os_log5(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@ %@", [c m0], [C m1]);
  os_log_pack_send(buf);
}

// FIXME: Lifetime of GenString's return should be extended in this case too.

// CHECK-NEWPM: define void @test_builtin_os_log6(
// CHECK-NEWPM: %[[CALL:.*]] = tail call %{{.*}} @GenString()
// CHECK-NEWPM: %[[V0:.*]] = bitcast %[[V1]]* %[[CALL]] to i8*
// CHECK-NEWPM: %[[V1:.*]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %[[V0]])
// CHECK-NEWPM: tail call void @llvm.objc.release(i8* %[[V1]])
// CHECK-NEWPM: tail call void @os_log_pack_send(
// CHECK-NEWPM-NOT: call void @llvm.objc.release(

void test_builtin_os_log6(void *buf) {
  __builtin_os_log_format(buf, "capabilities: %@", (0, GenString()));
  os_log_pack_send(buf);
}

#endif
