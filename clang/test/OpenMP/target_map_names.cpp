// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -debug-info-kind=limited -emit-llvm %s -o - | FileCheck %s --check-prefix DEBUG
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK
#ifndef HEADER
#define HEADER

// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";d;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";i;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";i[1:23];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";p;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";p[1:24];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.i;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.s.f;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.p[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.ps->s.i;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.ps->ps;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.ps->ps->ps;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.ps->ps->s.f[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.ps->ps->s.f[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->i;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->s.f;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->p[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->ps->s.i;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->ps->ps;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->ps->ps->ps;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->ps->ps->s.f[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->ps->ps->s.f[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.f[:22];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.p[:33];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";ps->p[:33];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.s;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"

struct S1 {
    int i;
    float f[50];
};

struct S2 {
    int i;
    float f[50];
    S1 s;
    double *p;
    struct S2 *ps;
};

void foo() {
  double d;
  int i[100];
  float *p;

  S2 s;
  S2 *ps;

#pragma omp target map(d)
  { }
#pragma omp target map(i)
  { }
#pragma omp target map(i[1:23])
  { }
#pragma omp target map(p)
  { }
#pragma omp target map(p[1:24])
  { }
#pragma omp target map(s)
  { }
#pragma omp target map(s.i)
  { }
#pragma omp target map(s.s.f)
  { }
#pragma omp target map(s.p)
  { }
#pragma omp target map(to: s.p[:22])
  { }
#pragma omp target map(s.ps)
  { }
#pragma omp target map(from: s.ps->s.i)
  { }
#pragma omp target map(to: s.ps->ps)
  { }
#pragma omp target map(s.ps->ps->ps)
  { }
#pragma omp target map(to: s.ps->ps->s.f[:22])
  { }
#pragma omp target map(ps)
  { }
#pragma omp target map(ps->i)
  { }
#pragma omp target map(ps->s.f)
  { }
#pragma omp target map(from: ps->p)
  { }
#pragma omp target map(to: ps->p[:22])
  { }
#pragma omp target map(ps->ps)
  { }
#pragma omp target map(from: ps->ps->s.i)
  { }
#pragma omp target map(from: ps->ps->ps)
  { }
#pragma omp target map(ps->ps->ps->ps)
  { }
#pragma omp target map(to: ps->ps->ps->s.f[:22])
  { }
#pragma omp target map(to: s.f[:22]) map(from: s.p[:33])
  { }
#pragma omp target map(from: s.f[:22]) map(to: ps->p[:33])
  { }
#pragma omp target map(from: s.f[:22], s.s) map(to: ps->p[:33])
  { }
}

// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";B;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";unknown;unknown;0;0;;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";A;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";x;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";fn;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"
// DEBUG: @{{.+}} = private constant [7 x i8*] [i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0), i8* getelementptr inbounds ([{{[0-9]+}} x i8], [{{[0-9]+}} x i8]* @{{[0-9]+}}, i32 0, i32 0)]

void bar(int N) {
  double B[10];
  double A[N];
  double x;
  S1 s;
  auto fn = [&x]() { return x; };
#pragma omp target
  {
    (void)B;
    (void)A;
    (void)fn();
    (void)s.f;
  }
}

// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";t;{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"

#pragma omp declare target
double t;
#pragma omp end declare target

void baz() {
#pragma omp target map(to:t)
  { }
#pragma omp target map(to:t) nowait
  { }
#pragma omp target teams map(to:t)
  { }
#pragma omp target teams map(to:t) nowait
  { }
#pragma omp target data map(to:t)
  { }
#pragma omp target enter data map(to:t)

#pragma omp target enter data map(to:t) nowait

#pragma omp target exit data map(from:t)

#pragma omp target exit data map(from:t) nowait

#pragma omp target update from(t)

#pragma omp target update to(t)

#pragma omp target update from(t) nowait

#pragma omp target update to(t) nowait
}

struct S3 {
  double Z[64];
};

#pragma omp declare mapper(id: S3 s) map(s.Z[0:64])

void qux() {
  S3 s;
#pragma omp target map(mapper(id), to:s)
  { }
}

// DEBUG: @{{[0-9]+}} = private unnamed_addr constant [{{[0-9]+}} x i8] c";s.Z[0:64];{{.*}}.cpp;{{[0-9]+}};{{[0-9]+}};;\00"

// DEBUG: %{{.+}} = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: %{{.+}} = call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}}, i32 {{.+}}, i32 {{.+}})
// DEBUG: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: %{{.+}} = call i32 @__tgt_target_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* %{{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: %{{.+}} = call i32 @__tgt_target_teams_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}}, i32 {{.+}}, i32 {{.+}})
// DEBUG: call void @__tgt_target_data_begin_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: call void @__tgt_target_data_end_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})
// DEBUG: call void @__tgt_target_data_update_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** getelementptr inbounds ([{{[0-9]+}} x i8*], [{{[0-9]+}} x i8*]* @.offload_mapnames{{.*}}, i32 0, i32 0), i8** {{.+}})

// CHECK: %{{.+}} = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: %{{.+}} = call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}}, i32 {{.+}}, i32 {{.+}})
// CHECK: call void @__tgt_target_data_begin_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: call void @__tgt_target_data_end_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: %{{.+}} = call i32 @__tgt_target_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* %{{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: %{{.+}} = call i32 @__tgt_target_teams_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}}, i32 {{.+}}, i32 {{.+}})
// CHECK: call void @__tgt_target_data_begin_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: call void @__tgt_target_data_end_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})
// CHECK: call void @__tgt_target_data_update_nowait_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** %{{.+}}, i8** %{{.+}}, i64* {{.+}}, i64* {{.+}}, i8** null, i8** {{.+}})


// DEBUG: void @.omp_mapper._ZTS2S3.id(i8* {{.*}}, i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 {{.*}}, i8* [[NAME_ARG:%.+]])
// DEBUG: store i8* [[NAME_ARG]], i8** [[NAME_STACK:%.+]]
// DEBUG: [[MAPPER_NAME:%.+]] = load i8*, i8** [[NAME_STACK]]
// DEBUG: call void @__tgt_push_mapper_component(i8* %{{.*}}, i8* %{{.*}}, i8* %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i8* [[MAPPER_NAME]])

#endif
