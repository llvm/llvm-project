// RUN: %clang_cc1 %s -S -emit-llvm -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

//CHECK: @[[STR1:.*]] = private unnamed_addr constant [{{.*}} x i8] c"{{.*}}attr-annotate.cpp\00", section "llvm.metadata"
//CHECK: @[[STR2:.*]] = private unnamed_addr constant [4 x i8] c"abc\00", align 1
//CHECK: @[[STR:.*]] = private unnamed_addr constant [5 x i8] c"test\00", section "llvm.metadata"
//CHECK: @[[ARGS:.*]] = private unnamed_addr constant { %struct.Struct } { %struct.Struct { ptr @_ZN1AIjLj9EE2SVE, ptr getelementptr (i8, ptr @_ZN1AIjLj9EE2SVE, i64 4) } }, section "llvm.metadata"
//CHECK: @[[ARGS2:.*]] = private unnamed_addr constant { i32, ptr, i32 } { i32 9, ptr @[[STR2:.*]], i32 8 }, section "llvm.metadata"
//CHECK: @llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_ZN1AIjLj9EE5test2Ev, ptr @.str.6, ptr @.str.1, i32 24, ptr @[[ARGS]] }, { ptr, ptr, ptr, i32, ptr } { ptr @_ZN1AIjLj9EE4testILi8EEEvv, ptr @[[STR:.*]], ptr @[[STR1:.*]], i32 {{.*}}, ptr @[[ARGS2:.*]] }]

constexpr const char* str() {
  return "abc";
}

template<typename T>
struct Struct {
  T t1;
  T t2;
};

template<typename T, T V>
struct A {
  static constexpr const T SV[] = {V, V + 1};
  template <int I> __attribute__((annotate("test", V, str(), I))) void test() {}
  __attribute__((annotate("test", Struct<const T*>{&SV[0], &SV[1]}))) void test2() {}
};

void t() {
  A<unsigned, 9> a;
  a.test<8>();
  a.test2();
}

template<typename T, T V>
struct B {
template<typename T1, T1 V1>
struct foo {
  int v __attribute__((annotate("v_ann_0", str(), 90, V))) __attribute__((annotate("v_ann_1", V1)));
};
};

static B<int long, -1>::foo<unsigned, 9> gf;

// CHECK-LABEL: @main(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[ARGC_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[ARGV_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    [[F:%.*]] = alloca %"struct.B<int, 7>::foo", align 4
// CHECK-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// CHECK-NEXT:    store i32 [[ARGC:%.*]], ptr [[ARGC_ADDR]], align 4
// CHECK-NEXT:    store ptr [[ARGV:%.*]], ptr [[ARGV_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[ARGC_ADDR]], align 4
// CHECK-NEXT:    [[V:%.*]] = getelementptr inbounds %"struct.B<int, 7>::foo", ptr [[F]], i32 0, i32 0
// CHECK-NEXT:    [[TMP2:%.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr [[V]], ptr @.str, ptr @.str.1, i32 {{.*}}, ptr @.args)
// CHECK-NEXT:    [[TMP5:%.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr [[TMP2]], ptr @.str.3, ptr @.str.1, i32 {{.*}}, ptr @.args.4)
// CHECK-NEXT:    store i32 [[TMP0]], ptr [[TMP5]], align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[ARGC_ADDR]], align 4
// CHECK-NEXT:    [[TMP8:%.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr @_ZL2gf, ptr @.str, ptr @.str.1, i32 {{.*}}, ptr @.args.5)
// CHECK-NEXT:    [[TMP11:%.*]] = call ptr @llvm.ptr.annotation.p0.p0(ptr [[TMP8]], ptr @.str.3, ptr @.str.1, i32 {{.*}}, ptr @.args.4)
// CHECK-NEXT:    store i32 [[TMP7]], ptr [[TMP11]], align 4
// CHECK-NEXT:    ret i32 0
//
int main(int argc, char **argv) {
    B<int, 7>::foo<unsigned, 9> f;
    f.v = argc;
    gf.v = argc;
    return 0;
}
