// RUN: %clang_cc1 -triple x86_64-unk-unk -emit-llvm -Os -o %t %s
// RUN: FileCheck < %t %s

struct s0 {
  unsigned int x[2] __attribute__((packed));
};

struct s1 {
  unsigned int x[2] __attribute__((packed));
  unsigned int y;
  unsigned int z __attribute__((packed));
};

struct s2 {
  unsigned int x[2] __attribute__((packed));
  unsigned int y __attribute__((packed));
  unsigned int z __attribute__((packed));
};

struct __attribute__((packed)) s3 {
  unsigned int x[2];
  unsigned int y;
  unsigned int z;
};

// CHECK: @align0 ={{.*}} local_unnamed_addr global i32 1
int align0 = __alignof(struct s0);
// CHECK: @align1 ={{.*}} local_unnamed_addr global i32 4
int align1 = __alignof(struct s1);
// CHECK: @align2 ={{.*}} local_unnamed_addr global i32 1
int align2 = __alignof(struct s2);
// CHECK: @align3 ={{.*}} local_unnamed_addr global i32 1
int align3 = __alignof(struct s3);

// CHECK: @align0_x ={{.*}} local_unnamed_addr global i32 1
int align0_x = __alignof(((struct s0*) 0)->x);
//
// CHECK: @align1_x ={{.*}} local_unnamed_addr global i32 1
int align1_x = __alignof(((struct s1*) 0)->x);
// CHECK: @align2_x ={{.*}} local_unnamed_addr global i32 1
int align2_x = __alignof(((struct s2*) 0)->x);
// CHECK: @align3_x ={{.*}} local_unnamed_addr global i32 1
int align3_x = __alignof(((struct s3*) 0)->x);

// CHECK: @align0_x0 ={{.*}} local_unnamed_addr global i32 4
int align0_x0 = __alignof(((struct s0*) 0)->x[0]);
// CHECK: @align1_x0 ={{.*}} local_unnamed_addr global i32 4
int align1_x0 = __alignof(((struct s1*) 0)->x[0]);
// CHECK: @align2_x0 ={{.*}} local_unnamed_addr global i32 4
int align2_x0 = __alignof(((struct s2*) 0)->x[0]);
// CHECK: @align3_x0 ={{.*}} local_unnamed_addr global i32 4
int align3_x0 = __alignof(((struct s3*) 0)->x[0]);

// CHECK-LABEL: define{{.*}} i32 @f0_a
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f0_b
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
int f0_a(struct s0 *a) {
  return a->x[1];
}
int f0_b(struct s0 *a) {
  return *(a->x + 1);
}

// Note that 'y' still causes struct s1 to be four-byte aligned.

// Note that we are incompatible with GCC on this example.
// 
// CHECK-LABEL: define{{.*}} i32 @f1_a
// CHECK:   load i32, ptr %{{.*}}, align 4
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f1_b
// CHECK:   load i32, ptr %{{.*}}, align 4
// CHECK: }

// Note that we are incompatible with GCC on this example.
//
// CHECK-LABEL: define{{.*}} i32 @f1_c
// CHECK:   load i32, ptr %{{.*}}, align 4
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f1_d
// CHECK:   load i32, ptr %{{.*}}, align 4
// CHECK: }
int f1_a(struct s1 *a) {
  return a->x[1];
}
int f1_b(struct s1 *a) {
  return *(a->x + 1);
}
int f1_c(struct s1 *a) {
  return a->y;
}
int f1_d(struct s1 *a) {
  return a->z;
}

// CHECK-LABEL: define{{.*}} i32 @f2_a
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f2_b
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f2_c
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f2_d
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
int f2_a(struct s2 *a) {
  return a->x[1];
}
int f2_b(struct s2 *a) {
  return *(a->x + 1);
}
int f2_c(struct s2 *a) {
  return a->y;
}
int f2_d(struct s2 *a) {
  return a->z;
}

// CHECK-LABEL: define{{.*}} i32 @f3_a
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f3_b
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f3_c
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
// CHECK-LABEL: define{{.*}} i32 @f3_d
// CHECK:   load i32, ptr %{{.*}}, align 1
// CHECK: }
int f3_a(struct s3 *a) {
  return a->x[1];
}
int f3_b(struct s3 *a) {
  return *(a->x + 1);
}
int f3_c(struct s3 *a) {
  return a->y;
}
int f3_d(struct s3 *a) {
  return a->z;
}

// Verify we don't claim things are overaligned.
//
// CHECK-LABEL: define{{.*}} double @f4
// CHECK:   load double, ptr {{.*}}, align 8
// CHECK: }
extern double g4[5] __attribute__((aligned(16)));
double f4(void) {
  return g4[1];
}
