// RUN: %clang                        -target x86_64 -O2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-NO-STRICT %s
// RUN: %clang -fstrict-flex-arrays=0 -target x86_64 -O2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-STRICT-0 %s
// RUN: %clang -fstrict-flex-arrays=1 -target x86_64 -O2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-STRICT-1 %s
// RUN: %clang -fstrict-flex-arrays=2 -target x86_64 -O2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-STRICT-2 %s
// RUN: %clang -fstrict-flex-arrays=3 -target x86_64 -O2 -S -emit-llvm %s -o - 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-STRICT-3 %s

#define OBJECT_SIZE_BUILTIN __builtin_object_size

typedef struct {
  float f;
  double c[];
} foo_t;

typedef struct {
  float f;
  double c[0];
} foo0_t;

typedef struct {
  float f;
  double c[1];
} foo1_t;

typedef struct {
  float f;
  double c[2];
} foo2_t;

// CHECK-LABEL: @bar(
unsigned bar(foo_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 -1
  // CHECK-STRICT-2: ret i32 -1
  // CHECK-STRICT-3: ret i32 -1
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @bar0(
unsigned bar0(foo0_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 -1
  // CHECK-STRICT-2: ret i32 -1
  // CHECK-STRICT-3: ret i32 0
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @bar1(
unsigned bar1(foo1_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 -1
  // CHECK-STRICT-2: ret i32 8
  // CHECK-STRICT-3: ret i32 8
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @bar2(
unsigned bar2(foo2_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 16
  // CHECK-STRICT-2: ret i32 16
  // CHECK-STRICT-3: ret i32 16
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

#define DYNAMIC_OBJECT_SIZE_BUILTIN __builtin_dynamic_object_size

// CHECK-LABEL: @dyn_bar(
unsigned dyn_bar(foo_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 -1
  // CHECK-STRICT-2: ret i32 -1
  // CHECK-STRICT-3: ret i32 -1
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @dyn_bar0(
unsigned dyn_bar0(foo0_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 -1
  // CHECK-STRICT-2: ret i32 -1
  // CHECK-STRICT-3: ret i32 0
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @dyn_bar1(
unsigned dyn_bar1(foo1_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 -1
  // CHECK-STRICT-2: ret i32 8
  // CHECK-STRICT-3: ret i32 8
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @dyn_bar2(
unsigned dyn_bar2(foo2_t *f) {
  // CHECK-NO-STRICT: ret i32 -1
  // CHECK-STRICT-0: ret i32 -1
  // CHECK-STRICT-1: ret i32 16
  // CHECK-STRICT-2: ret i32 16
  // CHECK-STRICT-3: ret i32 16
  return DYNAMIC_OBJECT_SIZE_BUILTIN(f->c, 1);
}

// Also checks for non-trailing flex-array like members

typedef struct {
  double c[0];
  float f;
} foofoo0_t;

typedef struct {
  double c[1];
  float f;
} foofoo1_t;

typedef struct {
  double c[2];
  float f;
} foofoo2_t;

// CHECK-LABEL: @babar0(
unsigned babar0(foofoo0_t *f) {
  // CHECK-NO-STRICT: ret i32 0
  // CHECK-STRICT-0: ret i32 0
  // CHECK-STRICT-1: ret i32 0
  // CHECK-STRICT-2: ret i32 0
  // CHECK-STRICT-3: ret i32 0
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @babar1(
unsigned babar1(foofoo1_t *f) {
  // CHECK-NO-STRICT: ret i32 8
  // CHECK-STRICT-0: ret i32 8
  // CHECK-STRICT-1: ret i32 8
  // CHECK-STRICT-2: ret i32 8
  // CHECK-STRICT-3: ret i32 8
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}

// CHECK-LABEL: @babar2(
unsigned babar2(foofoo2_t *f) {
  // CHECK-NO-STRICT: ret i32 16
  // CHECK-STRICT-0: ret i32 16
  // CHECK-STRICT-1: ret i32 16
  // CHECK-STRICT-2: ret i32 16
  // CHECK-STRICT-3: ret i32 16
  return OBJECT_SIZE_BUILTIN(f->c, 1);
}
