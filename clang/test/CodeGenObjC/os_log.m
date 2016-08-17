// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple -fobjc-arc -O2 | FileCheck %s

// Make sure we emit clang.arc.use before calling objc_release as part of the
// cleanup. This way we make sure the object will not be released until the
// end of the full expression.

// rdar://problem/24528966

@class NSString;
extern __attribute__((visibility("default"))) NSString * GenString();

// Behavior of __builtin_os_log differs between platforms, so only test on X86
#ifdef __x86_64__
// CHECK-LABEL: define i8* @test_builtin_os_log
void *test_builtin_os_log(void *buf) {
  return __builtin_os_log_format(buf, "capabilities: %@", GenString());

  // CHECK: store i8 2, i8*
  // CHECK: [[NUM_ARGS:%.*]] = getelementptr i8, i8* {{.*}}, i64 1
  // CHECK: store i8 1, i8* [[NUM_ARGS]]
  //
  // CHECK: [[ARG1_DESC:%.*]] = getelementptr i8, i8* {{.*}}, i64 2
  // CHECK: store i8 64, i8* [[ARG1_DESC]]
  // CHECK: [[ARG1_SIZE:%.*]] = getelementptr i8, i8* {{.*}}, i64 3
  // CHECK: store i8 8, i8* [[ARG1_SIZE]]
  // CHECK: [[ARG1:%.*]] = getelementptr i8, i8* {{.*}}, i64 4
  // CHECK: [[ARG1_CAST:%.*]] = bitcast i8* [[ARG1]] to

  // CHECK: [[STRING:%.*]] = {{.*}} call {{.*}} @GenString()
  // CHECK: [[STRING_CAST:%.*]] = bitcast {{.*}} [[STRING]] to
  // CHECK: call {{.*}} @objc_retainAutoreleasedReturnValue(i8* [[STRING_CAST]])
  // CHECK: store {{.*}} [[STRING]], {{.*}} [[ARG1_CAST]]

  // CHECK: call void (...) @clang.arc.use({{.*}} [[STRING]])
  // CHECK: call void @objc_release(i8* [[STRING_CAST]])
  // CHECK: ret i8*
}

#endif
