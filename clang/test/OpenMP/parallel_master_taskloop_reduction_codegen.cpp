// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - -triple powerpc64le-unknown-linux-gnu -std=c++98 -fnoopenmp-use-tls | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - -triple powerpc64le-unknown-linux-gnu -std=c++98 -fnoopenmp-use-tls | FileCheck --check-prefix SIMD-ONLY0 %s
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

#pragma omp parallel master taskloop reduction(+:sum, c[:n], d, e)
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
// CHECK:    [[ARGV_ADDR:%.*]] = alloca i8**,
// CHECK:    [[I:%.*]] = alloca i32,
// CHECK:    [[N:%.*]] = alloca i32,
// CHECK:    [[A:%.*]] = alloca [100 x float],
// CHECK:    [[B:%.*]] = alloca [100 x float],
// CHECK:    [[SUM:%.*]] = alloca float,
// CHECK:    [[SAVED_STACK:%.*]] = alloca i8*,
// CHECK:    [[C:%.*]] = alloca [100 x %struct.S],
// CHECK:    [[D:%.*]] = alloca float*,
// CHECK:    store i32 0, i32* [[RETVAL]],
// CHECK:    store i32 [[ARGC:%.*]], i32* [[ARGC_ADDR]],
// CHECK:    store i8** [[ARGV:%.*]], i8*** [[ARGV_ADDR]],
// CHECK:    [[TMP1:%.*]] = load i32, i32* [[ARGC_ADDR]],
// CHECK:    [[ADD:%.*]] = add nsw i32 [[TMP1]], 100
// CHECK:    [[TMP2:%.*]] = zext i32 [[ADD]] to i64
// CHECK:    [[VLA:%.+]] = alloca float, i64 %

// CHECK:    [[SUM_ADDR:%.*]] = alloca float*,
// CHECK:    [[AGG_CAPTURED:%.*]] = alloca [[STRUCT_ANON:%.*]],
// CHECK:    [[DOTRD_INPUT_:%.*]] = alloca [4 x %struct.kmp_taskred_input_t],
// CHECK:    alloca i32,
// CHECK:    [[DOTCAPTURE_EXPR_:%.*]] = alloca i32,
// CHECK:    [[DOTCAPTURE_EXPR_9:%.*]] = alloca i32,
// CHECK:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(
// CHECK-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:[^,]+]]
// CHECK:       [[THEN]]
// CHECK:    call void @__kmpc_taskgroup(%struct.ident_t*
// CHECK-DAG:    [[TMP21:%.*]] = bitcast float* %{{.+}} to i8*
// CHECK-DAG:    store i8* [[TMP21]], i8** [[TMP20:%[^,]+]],
// CHECK-DAG:    [[TMP20]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_:%.+]], i32 0, i32 0
// CHECK-DAG:    [[TMP21:%.*]] = bitcast float* %{{.+}} to i8*
// CHECK-DAG:    store i8* [[TMP21]], i8** [[TMP20:%[^,]+]],
// CHECK-DAG:    [[TMP20]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_:%.+]], i32 0, i32 1
// CHECK-DAG:    [[TMP22:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 2
// CHECK-DAG:    store i64 4, i64* [[TMP22]],
// CHECK-DAG:    [[TMP23:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 3
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_INIT1:.+]] to i8*), i8** [[TMP23]],
// CHECK-DAG:    [[TMP24:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 4
// CHECK-DAG:    store i8* null, i8** [[TMP24]],
// CHECK-DAG:    [[TMP25:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 5
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_COMB1:.+]] to i8*), i8** [[TMP25]],
// CHECK-DAG:    [[TMP26:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_]], i32 0, i32 6
// CHECK-DAG:    [[TMP27:%.*]] = bitcast i32* [[TMP26]] to i8*
// CHECK-DAG:    call void @llvm.memset.p0i8.i64(i8* align 8 [[TMP27]], i8 0, i64 4, i1 false)
// CHECK-DAG:    [[ARRAYIDX5:%.*]] = getelementptr inbounds [100 x %struct.S], [100 x %struct.S]* [[C:%.+]], i64 0, i64 0
// CHECK-DAG:    [[LB_ADD_LEN:%.*]] = add nsw i64 -1, %
// CHECK-DAG:    [[ARRAYIDX6:%.*]] = getelementptr inbounds [100 x %struct.S], [100 x %struct.S]* [[C]], i64 0, i64 [[LB_ADD_LEN]]
// CHECK-DAG:    [[TMP31:%.*]] = bitcast %struct.S* [[ARRAYIDX5]] to i8*
// CHECK-DAG:    store i8* [[TMP31]], i8** [[TMP28:%[^,]+]],
// CHECK-DAG:    [[TMP28]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4:%.+]], i32 0, i32 0
// CHECK-DAG:    [[TMP31:%.*]] = bitcast %struct.S* [[ARRAYIDX5]] to i8*
// CHECK-DAG:    store i8* [[TMP31]], i8** [[TMP28:%[^,]+]],
// CHECK-DAG:    [[TMP28]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4:%.+]], i32 0, i32 1
// CHECK-DAG:    [[TMP32:%.*]] = ptrtoint %struct.S* [[ARRAYIDX6]] to i64
// CHECK-DAG:    [[TMP33:%.*]] = ptrtoint %struct.S* [[ARRAYIDX5]] to i64
// CHECK-DAG:    [[TMP34:%.*]] = sub i64 [[TMP32]], [[TMP33]]
// CHECK-DAG:    [[TMP35:%.*]] = sdiv exact i64 [[TMP34]], ptrtoint (%struct.S* getelementptr (%struct.S, %struct.S* null, i32 1) to i64)
// CHECK-DAG:    [[TMP36:%.*]] = add nuw i64 [[TMP35]], 1
// CHECK-DAG:    [[TMP37:%.*]] = mul nuw i64 [[TMP36]], ptrtoint (%struct.S* getelementptr (%struct.S, %struct.S* null, i32 1) to i64)
// CHECK-DAG:    store i64 [[TMP37]], i64* [[TMP38:%[^,]+]],
// CHECK-DAG:    [[TMP38]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 2
// CHECK-DAG:    [[TMP39:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 3
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_INIT2:.+]] to i8*), i8** [[TMP39]],
// CHECK-DAG:    [[TMP40:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 4
// CHECK-DAG:    store i8* bitcast (void (i8*)* @[[RED_FINI2:.+]] to i8*), i8** [[TMP40]],
// CHECK-DAG:    [[TMP41:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 5
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_COMB2:.+]] to i8*), i8** [[TMP41]],
// CHECK-DAG:    [[TMP42:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_4]], i32 0, i32 6
// CHECK-DAG:    store i32 1, i32* [[TMP42]],
// CHECK-DAG:    [[TMP44:%.*]] = load float*, float** [[D:%.+]],
// CHECK-DAG:    [[TMP45:%.*]] = bitcast float* [[TMP44]] to i8*
// CHECK-DAG:    store i8* [[TMP45]], i8** [[TMP43:%[^,]+]],
// CHECK-DAG:    [[TMP43]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7:%.+]], i32 0, i32 0
// CHECK-DAG:    [[TMP45:%.*]] = bitcast float* [[TMP44]] to i8*
// CHECK-DAG:    store i8* [[TMP45]], i8** [[TMP43:%[^,]+]],
// CHECK-DAG:    [[TMP43]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7:%.+]], i32 0, i32 1
// CHECK-DAG:    [[TMP46:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 2
// CHECK-DAG:    store i64 4, i64* [[TMP46]],
// CHECK-DAG:    [[TMP47:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 3
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_INIT3:.+]] to i8*), i8** [[TMP47]],
// CHECK-DAG:    [[TMP48:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 4
// CHECK-DAG:    store i8* null, i8** [[TMP48]],
// CHECK-DAG:    [[TMP49:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 5
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_COMB3:.+]] to i8*), i8** [[TMP49]],
// CHECK-DAG:    [[TMP50:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_7]], i32 0, i32 6
// CHECK-DAG:    [[TMP51:%.*]] = bitcast i32* [[TMP50]] to i8*
// CHECK-DAG:    call void @llvm.memset.p0i8.i64(i8* align 8 [[TMP51]], i8 0, i64 4, i1 false)
// CHECK-DAG:    [[TMP53:%.*]] = bitcast float* [[VLA:%.+]] to i8*
// CHECK-DAG:    store i8* [[TMP53]], i8** [[TMP52:%[^,]+]],
// CHECK-DAG:    [[TMP52]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8:%.+]], i32 0, i32 0
// CHECK-DAG:    [[TMP53:%.*]] = bitcast float* [[VLA:%.+]] to i8*
// CHECK-DAG:    store i8* [[TMP53]], i8** [[TMP52:%[^,]+]],
// CHECK-DAG:    [[TMP52]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8:%.+]], i32 0, i32 1
// CHECK-DAG:    [[TMP54:%.*]] = mul nuw i64 [[TMP2:%.+]], 4
// CHECK-DAG:    [[TMP55:%.*]] = udiv exact i64 [[TMP54]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK-DAG:    store i64 [[TMP54]], i64* [[TMP56:%[^,]+]],
// CHECK-DAG:    [[TMP56]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 2
// CHECK-DAG:    [[TMP57:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 3
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_INIT4:.+]] to i8*), i8** [[TMP57]],
// CHECK-DAG:    [[TMP58:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 4
// CHECK-DAG:    store i8* null, i8** [[TMP58]],
// CHECK-DAG:    [[TMP59:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 5
// CHECK-DAG:    store i8* bitcast (void (i8*, i8*)* @[[RED_COMB4:.+]] to i8*), i8** [[TMP59]],
// CHECK-DAG:    [[TMP60:%.*]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[DOTRD_INPUT_GEP_8]], i32 0, i32 6
// CHECK-DAG:    store i32 1, i32* [[TMP60]],
// CHECK-DAG:    [[DOTRD_INPUT_GEP_]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], [4 x %struct.kmp_taskred_input_t]* [[DOTRD_INPUT_]], i64 0, i64
// CHECK-DAG:    [[DOTRD_INPUT_GEP_4]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], [4 x %struct.kmp_taskred_input_t]* [[DOTRD_INPUT_]], i64 0, i64
// CHECK-DAG:    [[DOTRD_INPUT_GEP_7]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], [4 x %struct.kmp_taskred_input_t]* [[DOTRD_INPUT_]], i64 0, i64
// CHECK-DAG:    [[DOTRD_INPUT_GEP_8]] = getelementptr inbounds [4 x %struct.kmp_taskred_input_t], [4 x %struct.kmp_taskred_input_t]* [[DOTRD_INPUT_]], i64 0, i64
// CHECK:    [[TMP61:%.*]] = bitcast [4 x %struct.kmp_taskred_input_t]* [[DOTRD_INPUT_]] to i8*
// CHECK:    [[TMP62:%.*]] = call i8* @__kmpc_taskred_init(i32 [[TMP0:%.+]], i32 4, i8* [[TMP61]])
// CHECK:    [[TMP63:%.*]] = load i32, i32* [[N:%.+]],
// CHECK:    store i32 [[TMP63]], i32* [[DOTCAPTURE_EXPR_]],
// CHECK:    [[TMP64:%.*]] = load i32, i32* [[DOTCAPTURE_EXPR_]],
// CHECK:    [[SUB:%.*]] = sub nsw i32 [[TMP64]], 0
// CHECK:    [[DIV:%.*]] = sdiv i32 [[SUB]], 1
// CHECK:    [[SUB12:%.*]] = sub nsw i32 [[DIV]], 1
// CHECK:    store i32 [[SUB12]], i32* [[DOTCAPTURE_EXPR_9]],
// CHECK:    [[TMP65:%.*]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* {{.+}}, i32 [[TMP0]], i32 1, i64 888, i64 40, i32 (i32, i8*)* bitcast (i32 (i32, %struct.kmp_task_t_with_privates*)* @[[TASK:.+]] to i32 (i32, i8*)*))
// CHECK:    call void @__kmpc_taskloop(%struct.ident_t* {{.+}}, i32 [[TMP0]], i8* [[TMP65]], i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* null)
// CHECK:    call void @__kmpc_end_taskgroup(%struct.ident_t*
// CHECK:  call {{.*}}void @__kmpc_end_master(
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]

// CHECK: define internal void @[[RED_INIT1]](i8* noalias noundef %{{.+}}, i8* noalias noundef %{{.+}})
// CHECK: store float 0.000000e+00, float* %
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB1]](i8* noundef %0, i8* noundef %1)
// CHECK: fadd float %
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

// CHECK: define internal void @[[RED_INIT2]](i8* noalias noundef %{{.+}}, i8* noalias noundef %{{.+}})
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call void [[OMP_INIT1:@.+]](
// CHECK: ret void

// CHECK: define internal void [[OMP_COMB1:@.+]](%struct.S* noalias noundef %0, %struct.S* noalias noundef %1)
// CHECK: fadd float %

// CHECK: define internal void [[OMP_INIT1]](%struct.S* noalias noundef %0, %struct.S* noalias noundef %1)
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

// CHECK: define internal void @[[RED_FINI2]](i8* noundef %0)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call void @
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB2]](i8* noundef %0, i8* noundef %1)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call void [[OMP_COMB1]](
// CHECK: ret void

// CHECK: define internal void @[[RED_INIT3]](i8* noalias noundef %{{.+}}, i8* noalias noundef %{{.+}})
// CHECK: store float 0.000000e+00, float* %
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB3]](i8* noundef %0, i8* noundef %1)
// CHECK: fadd float %
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

// CHECK: define internal void @[[RED_INIT4]](i8* noalias noundef %{{.+}}, i8* noalias noundef %{{.+}})
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: store float 0.000000e+00, float* %
// CHECK: ret void

// CHECK: define internal void @[[RED_COMB4]](i8* noundef %0, i8* noundef %1)
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: fadd float %
// CHECK: store float %{{.+}}, float* %
// CHECK: ret void

// CHECK-NOT: call i8* @__kmpc_threadprivate_cached(
// CHECK: call i8* @__kmpc_task_reduction_get_th_data(
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call i8* @__kmpc_task_reduction_get_th_data(
// CHECK-NOT: call i8* @__kmpc_threadprivate_cached(
// CHECK: call i8* @__kmpc_task_reduction_get_th_data(
// CHECK: call i8* @__kmpc_threadprivate_cached(
// CHECK: call i8* @__kmpc_task_reduction_get_th_data(
// CHECK-NOT: call i8* @__kmpc_threadprivate_cached(

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
