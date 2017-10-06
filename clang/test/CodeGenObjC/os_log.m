// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O2 | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O0 | FileCheck %s -check-prefix=CHECK-O0

// Make sure we emit clang.arc.use before calling objc_release as part of the
// cleanup. This way we make sure the object will not be released until the
// end of the full expression.

// rdar://problem/24528966

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
  // CHECK: %[[V1:.*]] = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %[[V0]])
  // CHECK: %[[V2:.*]] = ptrtoint %[[TY0]]* %[[CALL]] to i64
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
  // CHECK: tail call void (...) @clang.arc.use(%[[TY0]]* %[[CALL]])
  // CHECK: tail call void @objc_release(i8* %[[V0]])
  // CHECK: ret i8* %[[BUF]]

  // clang.arc.use is used and removed in IR optimizations. At O0, we should not
  // emit clang.arc.use, since it will not be removed and we will have a link
  // error.
  // CHECK-O0: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK-O0: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK-O0: %[[V0:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK-O0: %[[CALL:.*]] = call %[[TY0:.*]]* (...) @GenString()
  // CHECK-O0: %[[V1:.*]] = bitcast %[[TY0]]* %[[CALL]] to i8*
  // CHECK-O0: %[[V2:.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* %[[V1]])
  // CHECK-O0: %[[V3:.*]] = bitcast i8* %[[V2]] to %[[TY0]]*
  // CHECK-O0: %[[V4:.*]] = ptrtoint %[[TY0]]* %[[V3]] to i64
  // CHECK-O0: call void @__os_log_helper_1_2_1_8_64(i8* %[[V0]], i64 %[[V4]])
  // CHECK-O0: %[[V5:.*]] = bitcast %[[TY0]]* %[[V3]] to i8*
  // CHECK-O0-NOT call void (...) @clang.arc.use({{.*}}
  // CHECK-O0: call void @objc_release(i8* %[[V5]])
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

#endif
