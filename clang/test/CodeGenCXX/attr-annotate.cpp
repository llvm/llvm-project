// RUN: %clang_cc1 %s -S -emit-llvm -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

//CHECK: @[[STR1:.*]] = private unnamed_addr constant [{{.*}} x i8] c"{{.*}}attr-annotate.cpp\00", section "llvm.metadata"
//CHECK: @[[STR2:.*]] = private unnamed_addr constant [4 x i8] c"abc\00", align 1
//CHECK: @[[STR:.*]] = private unnamed_addr constant [5 x i8] c"test\00", section "llvm.metadata"
//CHECK: @[[ARGS:.*]] = private unnamed_addr constant { i32, i8*, i32 } { i32 9, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR2:.*]], i32 0, i32 0), i32 8 }, section "llvm.metadata"
//CHECK: @[[ARGS2:.*]] = private unnamed_addr constant { %struct.Struct } { %struct.Struct { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZN1AIjLj9EE2SVE, i32 0, i32 0), i32* bitcast (i8* getelementptr (i8, i8* bitcast ([2 x i32]* @_ZN1AIjLj9EE2SVE to i8*), i64 4) to i32*) } }, section "llvm.metadata"
//CHECK: @llvm.global.annotations = appending global [2 x { i8*, i8*, i8*, i32, i8* }] [{ i8*, i8*, i8*, i32, i8* } { i8* bitcast (void (%struct.A*)* @_ZN1AIjLj9EE4testILi8EEEvv to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @[[STR:.*]], i32 0, i32 0), i8* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8]* @[[STR1:.*]], i32 0, i32 0), i32 {{.*}}, i8* bitcast ({ i32, i8*, i32 }* @[[ARGS:.*]] to i8*) }, { i8*, i8*, i8*, i32, i8* } { i8* bitcast (void (%struct.A*)* @_ZN1AIjLj9EE5test2Ev to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8]* @.str.1, i32 0, i32 0), i32 24, i8* bitcast ({ %struct.Struct }* @[[ARGS2]] to i8*) }]

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
// CHECK-NEXT:    [[ARGV_ADDR:%.*]] = alloca i8**, align 8
// CHECK-NEXT:    [[F:%.*]] = alloca %"struct.B<int, 7>::foo", align 4
// CHECK-NEXT:    store i32 0, i32* [[RETVAL]], align 4
// CHECK-NEXT:    store i32 [[ARGC:%.*]], i32* [[ARGC_ADDR]], align 4
// CHECK-NEXT:    store i8** [[ARGV:%.*]], i8*** [[ARGV_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* [[ARGC_ADDR]], align 4
// CHECK-NEXT:    [[V:%.*]] = getelementptr inbounds %"struct.B<int, 7>::foo", %"struct.B<int, 7>::foo"* [[F]], i32 0, i32 0
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast i32* [[V]] to i8*
// CHECK-NEXT:    [[TMP2:%.*]] = call i8* @llvm.ptr.annotation.p0i8(i8* [[TMP1]], i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8]* @.str.1, i32 0, i32 0), i32 {{.*}}, i8* bitcast ({ i8*, i32, i32 }* @.args to i8*))
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast i8* [[TMP2]] to i32*
// CHECK-NEXT:    [[TMP4:%.*]] = bitcast i32* [[TMP3]] to i8*
// CHECK-NEXT:    [[TMP5:%.*]] = call i8* @llvm.ptr.annotation.p0i8(i8* [[TMP4]], i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8]* @.str.1, i32 0, i32 0), i32 {{.*}}, i8* bitcast ({ i32 }* @.args.4 to i8*))
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast i8* [[TMP5]] to i32*
// CHECK-NEXT:    store i32 [[TMP0]], i32* [[TMP6]], align 4
// CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[ARGC_ADDR]], align 4
// CHECK-NEXT:    [[TMP8:%.*]] = call i8* @llvm.ptr.annotation.p0i8(i8* bitcast (%"struct.B<long, -1>::foo"* @_ZL2gf to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8]* @.str.1, i32 0, i32 0), i32 {{.*}}, i8* bitcast ({ i8*, i32, i64 }* @.args.5 to i8*))
// CHECK-NEXT:    [[TMP9:%.*]] = bitcast i8* [[TMP8]] to i32*
// CHECK-NEXT:    [[TMP10:%.*]] = bitcast i32* [[TMP9]] to i8*
// CHECK-NEXT:    [[TMP11:%.*]] = call i8* @llvm.ptr.annotation.p0i8(i8* [[TMP10]], i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([{{.*}} x i8], [{{.*}} x i8]* @.str.1, i32 0, i32 0), i32 {{.*}}, i8* bitcast ({ i32 }* @.args.4 to i8*))
// CHECK-NEXT:    [[TMP12:%.*]] = bitcast i8* [[TMP11]] to i32*
// CHECK-NEXT:    store i32 [[TMP7]], i32* [[TMP12]], align 4
// CHECK-NEXT:    ret i32 0
//
int main(int argc, char **argv) {
    B<int, 7>::foo<unsigned, 9> f;
    f.v = argc;
    gf.v = argc;
    return 0;
}
