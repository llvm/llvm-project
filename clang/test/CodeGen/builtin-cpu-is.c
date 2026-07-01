// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

// CHECK: @__cpu_model = external dso_local global { i32, i32, i32, [1 x i32] }

void intel(void) {
  if (__builtin_cpu_is("intel"))
    a("intel");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void amd(void) {
  if (__builtin_cpu_is("amd"))
    a("amd");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
  // CHECK: = icmp eq i32 [[LOAD]], 2
}

void atom(void) {
  if (__builtin_cpu_is("atom"))
    a("atom");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void amdfam10h(void) {
  if (__builtin_cpu_is("amdfam10h"))
    a("amdfam10h");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void barcelona(void) {
  if (__builtin_cpu_is("barcelona"))
    a("barcelona");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

void nehalem(void) {
  if (__builtin_cpu_is("nehalem"))
    a("nehalem");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 1
}

void other(void) {
  if (__builtin_cpu_is("other"))
    a("other");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr @__cpu_model
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

// A type alias must map to the same ABI value as its canonical entry.
void slm(void) {
  if (__builtin_cpu_is("slm"))
    a("slm");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 6
}

void amdfam10(void) {
  if (__builtin_cpu_is("amdfam10"))
    a("amdfam10");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 4)
  // CHECK: = icmp eq i32 [[LOAD]], 4
}

// Subtype ABI values below have gaps reserved for Zhaoxin CPUs; they must
// match the values implemented in compiler-rt/libgcc exactly.
void znver5(void) {
  if (__builtin_cpu_is("znver5"))
    a("znver5");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 36
}

void diamondrapids(void) {
  if (__builtin_cpu_is("diamondrapids"))
    a("diamondrapids");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 38
}

void novalake(void) {
  if (__builtin_cpu_is("novalake"))
    a("novalake");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 39
}

void znver6(void) {
  if (__builtin_cpu_is("znver6"))
    a("znver6");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 40
}

void c86_4g_m4(void) {
  if (__builtin_cpu_is("c86-4g-m4"))
    a("c86-4g-m4");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 41
}

void c86_4g_m8(void) {
  if (__builtin_cpu_is("c86-4g-m8"))
    a("c86-4g-m8");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 44
}

// Subtype aliases must map to the same ABI value as their canonical entry.
void raptorlake(void) {
  if (__builtin_cpu_is("raptorlake"))
    a("raptorlake");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void meteorlake(void) {
  if (__builtin_cpu_is("meteorlake"))
    a("meteorlake");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void gracemont(void) {
  if (__builtin_cpu_is("gracemont"))
    a("gracemont");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 25
}

void emeraldrapids(void) {
  if (__builtin_cpu_is("emeraldrapids"))
    a("emeraldrapids");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 24
}

void lunarlake(void) {
  if (__builtin_cpu_is("lunarlake"))
    a("lunarlake");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 33
}

void wildcatlake(void) {
  if (__builtin_cpu_is("wildcatlake"))
    a("wildcatlake");

  // CHECK: [[LOAD:%[^ ]+]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 8)
  // CHECK: = icmp eq i32 [[LOAD]], 34
}
