// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} void @{{.*}}test4{{.*}}(float
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP2:%.*]] = alloca float, align 4
// CHECK-NEXT:    store float [[TMP1:%.*]], float* [[TMP2:%.*]], align 4
// CHECK-NEXT:    ret void

// CHECK-LABEL: define{{.*}} void @{{.*}}test4{{.*}}(i32
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP2:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[TMP1:%.*]], i32* [[TMP2:%.*]], align 4
// CHECK-NEXT:    ret void

// CHECK-LABEL: define{{.*}} void @{{.*}}test{{.*}}(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP1:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP1:%.*]], align 4
// CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3:%.*]]
// CHECK-NEXT:    call void @{{.*}}test4{{.*}}(i32 noundef [[TMP4:%.*]])
// CHECK-NEXT:    [[TMP5:%.*]] = load float, float* [[TMP2:%.*]], align 4
// CHECK-NEXT:    [[TMP6:%.*]] = freeze float [[TMP5:%.*]]
// CHECK-NEXT:    call void @{{.*}}test4{{.*}}(float noundef [[TMP6:%.*]])
// CHECK-NEXT:    ret void

template<class T>
void test4(T __attribute__((maybe_undef)) arg) {
  return;
}

template
void test4<float>(float arg);

template
void test4<int>(int arg);

void test() {
    int Var1;
    float Var2;
    test4<int>(Var1);
    test4<float>(Var2);
}
