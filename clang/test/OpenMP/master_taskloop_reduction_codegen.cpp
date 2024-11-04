// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -Wno-vla -debug-info-kind=limited -emit-llvm -o - -triple powerpc64le-unknown-linux-gnu -fnoopenmp-use-tls -std=c++98 | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ %s -verify -Wno-vla -debug-info-kind=limited -emit-llvm -o - -triple powerpc64le-unknown-linux-gnu -fnoopenmp-use-tls -std=c++98 | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

struct S {
  float a;
  S() : a(0.0f) {}
  ~S() {}
};

#pragma omp declare reduction(+:S:omp_out.a += omp_in.a) initializer(omp_priv = omp_orig)

float g;

int a;
#pragma omp threadprivate(a)
int main (int argc, char *argv[])
{
int   i, n;
float a[100], b[100], sum, e[argc + 100];
S c[100];
float &d = g;

/* Some initializations */
n = 100;
for (i=0; i < n; i++)
  a[i] = b[i] = i * 1.0;
sum = 0.0;

#pragma omp master taskloop reduction(+:sum, c[:n], d, e)
  for (i=0; i < n; i++) {
    sum = sum + (a[i] * b[i]);
    c[i].a = i*i;
    d += i*i;
    e[i] = i;
  }

}

// CHECK-LABEL: @main(
// CHECK:    [[RETVAL:%.*]] = alloca i32,
// CHECK:    [[ARGC_ADDR:%.*]] = alloca i32,
// CHECK:    [[ARGV_ADDR:%.*]] = alloca ptr,
// CHECK:    [[I:%.*]] = alloca i32,
// CHECK:    [[N:%.*]] = alloca i32,
// CHECK:    [[A:%.*]] = alloca [100 x float],
// CHECK:    [[B:%.*]] = alloca [100 x float],
// CHECK:    [[SUM:%.*]] = alloca float,
// CHECK:    [[SAVED_STACK:%.*]] = alloca ptr,
// CHECK:    [[C:%.*]] = alloca [100 x %struct.S],
// CHECK:    [[D:%.*]] = alloca ptr,
// CHECK:    [[AGG_CAPTURED:%.*]] = alloca [[STRUCT_ANON:%.*]],
// CHECK:    [[DOTRD_INPUT_:%.*]] = alloca [4 x %struct.kmp_taskred_input_t],
// CHECK:    alloca i32,
// CHECK:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i32,
// CHECK:    [[DOTCAPTURE_EXPR_9:%.*]] = alloca i32,
// CHECK:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(ptr
// CHECK:    store i32 0, ptr [[RETVAL]],
// CHECK:    store i32 [[ARGC:%.*]], ptr [[ARGC_ADDR]],
// CHECK:    store ptr [[ARGV:%.*]], ptr [[ARGV_ADDR]],
// CHECK:    [[TMP1:%.*]] = load i32, ptr [[ARGC_ADDR]],
// CHECK:    [[ADD:%.*]] = add nsw i32 [[TMP1]], 100
// CHECK:    [[TMP2:%.*]] = zext i32 [[ADD]] to i64
// CHECK:    [[VLA:%.+]] = alloca float, i64 %

// CHECK:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(
// CHECK-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:[^,]+]]
// CHECK:       [[THEN]]
// CHECK:    call void @__kmpc_taskgroup(ptr
// CHECK-DAG:    store ptr [[SUM]], ptr [[TMP20:%[^,]+]],
// CHECK-DAG:    [[TMP20]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_:%.+]], i32 0, i32 0
// CHECK-DAG:    store ptr [[SUM]], ptr [[TMP20:%[^,]+]],
// CHECK-DAG:    [[TMP20]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 1
// CHECK-DAG:    [[TMP22:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 2
// CHECK-DAG:    store i64 4, ptr [[TMP22]],
// CHECK-DAG:    [[TMP23:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 3
// CHECK-DAG:    store ptr @[[RED_INIT1:.+]], ptr [[TMP23]],
// CHECK-DAG:    [[TMP24:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 4
// CHECK-DAG:    store ptr null, ptr [[TMP24]],
// CHECK-DAG:    [[TMP25:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 5
// CHECK-DAG:    store ptr @[[RED_COMB1:.+]], ptr [[TMP25]],
// CHECK-DAG:    [[TMP26:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_]], i32 0, i32 6
// CHECK-DAG:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP26]], i8 0, i64 4, i1 false)
// CHECK-DAG:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [100 x %struct.S], ptr [[C]], i64 0, i64 0
// CHECK-DAG:    [[LB_ADD_LEN:%.*]] = add nsw i64 -1, %
// CHECK-DAG:    [[ARRAYIDX6:%.*]] = getelementptr inbounds [100 x %struct.S], ptr [[C]], i64 0, i64 [[LB_ADD_LEN]]
// CHECK-DAG:    store ptr [[ARRAYIDX5]], ptr [[TMP28:%[^,]+]],
// CHECK-DAG:    [[TMP28]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4:%.+]], i32 0, i32 0
// CHECK-DAG:    store ptr [[ARRAYIDX5]], ptr [[TMP28:%[^,]+]],
// CHECK-DAG:    [[TMP28]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 1
// CHECK-DAG:    [[TMP32:%.*]] = ptrtoint ptr [[ARRAYIDX6]] to i64
// CHECK-DAG:    [[TMP33:%.*]] = ptrtoint ptr [[ARRAYIDX5]] to i64
// CHECK-DAG:    [[TMP34:%.*]] = sub i64 [[TMP32]], [[TMP33]]
// CHECK-DAG:    [[TMP35:%.*]] = sdiv exact i64 [[TMP34]], ptrtoint (ptr getelementptr (%struct.S, ptr null, i32 1) to i64)
// CHECK-DAG:    [[TMP36:%.*]] = add nuw i64 [[TMP35]], 1
// CHECK-DAG:    [[TMP37:%.*]] = mul nuw i64 [[TMP36]], ptrtoint (ptr getelementptr (%struct.S, ptr null, i32 1) to i64)
// CHECK-DAG:    store i64 [[TMP37]], ptr [[TMP38:%[^,]+]],
// CHECK-DAG:    [[TMP38]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 2
// CHECK-DAG:    [[TMP39:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 3
// CHECK-DAG:    store ptr @[[RED_INIT2:.+]], ptr [[TMP39]],
// CHECK-DAG:    [[TMP40:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 4
// CHECK-DAG:    store ptr @[[RED_FINI2:.+]], ptr [[TMP40]],
// CHECK-DAG:    [[TMP41:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 5
// CHECK-DAG:    store ptr @[[RED_COMB2:.+]], ptr [[TMP41]],
// CHECK-DAG:    [[TMP42:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_4]], i32 0, i32 6
// CHECK-DAG:    store i32 1, ptr [[TMP42]],
// CHECK-DAG:    [[TMP44:%.*]] = load ptr, ptr [[D]],
// CHECK-DAG:    store ptr [[TMP44]], ptr [[TMP43:%[^,]+]],
// CHECK-DAG:    [[TMP43]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7:%.+]], i32 0, i32 0
// CHECK-DAG:    store ptr [[TMP44]], ptr [[TMP43:%[^,]+]],
// CHECK-DAG:    [[TMP43]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7]], i32 0, i32 1
// CHECK-DAG:    [[TMP46:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7]], i32 0, i32 2
// CHECK-DAG:    store i64 4, ptr [[TMP46]],
// CHECK-DAG:    [[TMP47:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7]], i32 0, i32 3
// CHECK-DAG:    store ptr @[[RED_INIT3:.+]], ptr [[TMP47]],
// CHECK-DAG:    [[TMP48:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7]], i32 0, i32 4
// CHECK-DAG:    store ptr null, ptr [[TMP48]],
// CHECK-DAG:    [[TMP49:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7]], i32 0, i32 5
// CHECK-DAG:    store ptr @[[RED_COMB3:.+]], ptr [[TMP49]],
// CHECK-DAG:    [[TMP50:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_7]], i32 0, i32 6
// CHECK-DAG:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP50]], i8 0, i64 4, i1 false)
// CHECK-DAG:    store ptr [[VLA]], ptr [[TMP52:%[^,]+]],
// CHECK-DAG:    [[TMP52]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8:%.+]], i32 0, i32 0
// CHECK-DAG:    store ptr [[VLA]], ptr [[TMP52:%[^,]+]],
// CHECK-DAG:    [[TMP52]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8]], i32 0, i32 1
// CHECK-DAG:    [[TMP54:%.*]] = mul nuw i64 [[TMP2]], 4
// CHECK-DAG:    [[TMP55:%.*]] = udiv exact i64 [[TMP54]], ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
// CHECK-DAG:    store i64 [[TMP54]], ptr [[TMP56:%[^,]+]],
// CHECK-DAG:    [[TMP56]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8]], i32 0, i32 2
// CHECK-DAG:    [[TMP57:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8]], i32 0, i32 3
// CHECK-DAG:    store ptr @[[RED_INIT4:.+]], ptr [[TMP57]],
// CHECK-DAG:    [[TMP58:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8]], i32 0, i32 4
// CHECK-DAG:    store ptr null, ptr [[TMP58]],
// CHECK-DAG:    [[TMP59:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8]], i32 0, i32 5
// CHECK-DAG:    store ptr @[[RED_COMB4:.+]], ptr [[TMP59]],
// CHECK-DAG:    [[TMP60:%.*]] = getelementptr inbounds nuw %struct.kmp_taskred_input_t, ptr [[DOTRD_INPUT_GEP_8]], i32 0, i32 6
// CHECK-DAG:    store i32 1, ptr [[TMP60]],
// CHECK-DAG:    [[DOTRD_INPUT_GEP_]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64
// CHECK-DAG:    [[DOTRD_INPUT_GEP_4]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64
// CHECK-DAG:    [[DOTRD_INPUT_GEP_7]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64
// CHECK-DAG:    [[DOTRD_INPUT_GEP_8]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], ptr [[DOTRD_INPUT_]], i64 0, i64
// CHECK:    [[TMP62:%.*]] = call ptr @__kmpc_taskred_init(i32 [[TMP0]], i32 4, ptr [[DOTRD_INPUT_]])
// CHECK:    [[TMP63:%.*]] = load i32, ptr [[N]],
// CHECK:    store i32 [[TMP63]], ptr [[DOTCAPTURE_EXPR_]],
// CHECK:    [[TMP64:%.*]] = load i32, ptr [[DOTCAPTURE_EXPR_]],
// CHECK:    [[SUB:%.*]] = sub nsw i32 [[TMP64]], 0
// CHECK:    [[DIV:%.*]] = sdiv i32 [[SUB]], 1
// CHECK:    [[SUB12:%.*]] = sub nsw i32 [[DIV]], 1
// CHECK:    store i32 [[SUB12]], ptr [[DOTCAPTURE_EXPR_9]],
// CHECK:    [[TMP65:%.*]] = call ptr @__kmpc_omp_task_alloc(ptr {{.+}}, i32 [[TMP0]], i32 1, i64 888, i64 40, ptr @[[TASK:.+]])
// CHECK:    call void @__kmpc_taskloop(ptr {{.+}}, i32 [[TMP0]], ptr [[TMP65]], i32 1, ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, ptr null)
// CHECK:    call void @__kmpc_end_taskgroup(ptr
// CHECK:  call {{.*}}void @__kmpc_end_master(
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]

// CHECK:    ret i32

// CHECK: define internal void @[[RED_INIT1]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK: store float 0.000000e+00, ptr %
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB1]](ptr noundef %0, ptr noundef %1)
// CHECK: fadd float %
// CHECK: store float %{{.+}}, ptr %
// CHECK: ret void

// CHECK: define internal void @[[RED_INIT2]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK-NOT: call ptr @__kmpc_threadprivate_cached(
// CHECK: call void [[OMP_INIT1:@.+]](
// CHECK: ret void

// CHECK: define internal void [[OMP_COMB1:@.+]](ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: fadd float %

// CHECK: define internal void [[OMP_INIT1]](ptr noalias noundef %0, ptr noalias noundef %1)
// CHECK: call void @llvm.memcpy.p0.p0.i64(

// CHECK: define internal void @[[RED_FINI2]](ptr noundef %0)
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK: call void @
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB2]](ptr noundef %0, ptr noundef %1)
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK: call void [[OMP_COMB1]](
// CHECK: ret void

// CHECK: define internal void @[[RED_INIT3]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK: store float 0.000000e+00, ptr %
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB3]](ptr noundef %0, ptr noundef %1)
// CHECK: fadd float %
// CHECK: store float %{{.+}}, ptr %
// CHECK: ret void

// CHECK: define internal void @[[RED_INIT4]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK: store float 0.000000e+00, ptr %
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB4]](ptr noundef %0, ptr noundef %1)
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK: fadd float %
// CHECK: store float %{{.+}}, ptr %
// CHECK: ret void

// CHECK-NOT: call ptr @__kmpc_threadprivate_cached(
// CHECK: call ptr @__kmpc_task_reduction_get_th_data(
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK: call ptr @__kmpc_task_reduction_get_th_data(
// CHECK-NOT: call ptr @__kmpc_threadprivate_cached(
// CHECK: call ptr @__kmpc_task_reduction_get_th_data(
// CHECK: call ptr @__kmpc_threadprivate_cached(
// CHECK: call ptr @__kmpc_task_reduction_get_th_data(
// CHECK-NOT: call ptr @__kmpc_threadprivate_cached(

// CHECK-DAG: distinct !DISubprogram(linkageName: "[[TASK]]", scope: !
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_INIT1]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_COMB1]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_INIT2]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_FINI2]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_COMB2]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_INIT3]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_COMB3]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_INIT4]]"
// CHECK-DAG: !DISubprogram(linkageName: "[[RED_COMB4]]"
