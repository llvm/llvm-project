// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify while loop is recognized after sequence of pragma clang loop directives.
void while_test(int *List, int Length) {
  // CHECK: define {{.*}} @_Z10while_test
  int i = 0;

#pragma clang loop vectorize(enable)
#pragma clang loop interleave_count(4)
#pragma clang loop vectorize_width(4)
#pragma clang loop unroll(full)
#pragma clang loop distribute(enable)
  while (i < Length) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
    List[i] = i * 2;
    i++;
  }
}

// Verify do loop is recognized after multi-option pragma clang loop directive.
void do_test(int *List, int Length) {
  int i = 0;

#pragma clang loop vectorize_width(8) interleave_count(4) unroll(disable) distribute(disable)
  do {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
    List[i] = i * 2;
    i++;
  } while (i < Length);
}

enum struct Tuner : short { Interleave = 4, Unroll = 8 };

// Verify for loop is recognized after sequence of pragma clang loop directives.
void for_test(int *List, int Length) {
#pragma clang loop interleave(enable)
#pragma clang loop interleave_count(static_cast<int>(Tuner::Interleave))
#pragma clang loop unroll_count(static_cast<int>(Tuner::Unroll))
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
    List[i] = i * 2;
  }
}

// Verify c++11 for range loop is recognized after
// sequence of pragma clang loop directives.
void for_range_test() {
  double List[100];

#pragma clang loop vectorize_width(2) interleave_count(2)
  for (int i : List) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
    List[i] = i;
  }
}

// Verify disable pragma clang loop directive generates correct metadata
void disable_test(int *List, int Length) {
#pragma clang loop vectorize(disable) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_5:.*]]
    List[i] = i * 2;
  }
}

#define VECWIDTH 2
#define INTCOUNT 2
#define UNROLLCOUNT 8

// Verify defines are correctly resolved in pragma clang loop directive
void for_define_test(int *List, int Length, int Value) {
#pragma clang loop vectorize_width(VECWIDTH) interleave_count(INTCOUNT)
#pragma clang loop unroll_count(UNROLLCOUNT)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_6:.*]]
    List[i] = i * Value;
  }
}

// Verify constant expressions are handled correctly.
void for_contant_expression_test(int *List, int Length) {
#pragma clang loop vectorize_width(1 + 4)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_7:.*]]
    List[i] = i;
  }

#pragma clang loop vectorize_width(3 + VECWIDTH)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_8:.*]]
    List[i] += i;
  }
}

// Verify metadata is generated when template is used.
template <typename A>
void for_template_test(A *List, int Length, A Value) {
#pragma clang loop vectorize_width(8) interleave_count(8) unroll_count(8)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_9:.*]]
    List[i] = i * Value;
  }
}

// Verify define is resolved correctly when template is used.
template <typename A, typename T>
void for_template_define_test(A *List, int Length, A Value) {
  const T VWidth = VECWIDTH;
  const T ICount = INTCOUNT;
  const T UCount = UNROLLCOUNT;
#pragma clang loop vectorize_width(VWidth) interleave_count(ICount)
#pragma clang loop unroll_count(UCount)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_10:.*]]
    List[i] = i * Value;
  }
}

// Verify templates and constant expressions are handled correctly.
template <typename A, int V, int I, int U>
void for_template_constant_expression_test(A *List, int Length) {
#pragma clang loop vectorize_width(V) interleave_count(I) unroll_count(U)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_11:.*]]
    List[i] = i;
  }

#pragma clang loop vectorize_width(V * 2 + VECWIDTH) interleave_count(I * 2 + INTCOUNT) unroll_count(U * 2 + UNROLLCOUNT)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_12:.*]]
    List[i] += i;
  }

  const int Scale = 4;
#pragma clang loop vectorize_width(Scale * V) interleave_count(Scale * I) unroll_count(Scale * U)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_13:.*]]
    List[i] += i;
  }

#pragma clang loop vectorize_width((Scale * V) + 2)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_14:.*]]
    List[i] += i;
  }
}

#undef VECWIDTH
#undef INTCOUNT
#undef UNROLLCOUNT

// Use templates defined above. Test verifies metadata is generated correctly.
void template_test(double *List, int Length) {
  double Value = 10;

  for_template_test<double>(List, Length, Value);
  for_template_define_test<double, int>(List, Length, Value);
  for_template_constant_expression_test<double, 2, 4, 8>(List, Length);
}

// Verify for loop is performing fixed width vectorization
void for_test_fixed_16(int *List, int Length) {
#pragma clang loop vectorize_width(16, fixed) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_15:.*]]
    List[i] = i * 2;
  }
}

// Verify for loop is performing scalable vectorization
void for_test_scalable_16(int *List, int Length) {
#pragma clang loop vectorize_width(16, scalable) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_16:.*]]
    List[i] = i * 2;
  }
}

// Verify for loop is performing fixed width vectorization
void for_test_fixed(int *List, int Length) {
#pragma clang loop vectorize_width(fixed) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_17:.*]]
    List[i] = i * 2;
  }
}

// Verify for loop is performing scalable vectorization
void for_test_scalable(int *List, int Length) {
#pragma clang loop vectorize_width(scalable) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_18:.*]]
    List[i] = i * 2;
  }
}

// Verify for loop is performing scalable vectorization
void for_test_scalable_1(int *List, int Length) {
#pragma clang loop vectorize_width(1, scalable) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_19:.*]]
    List[i] = i * 2;
  }
}

// Verify for loop is not performing vectorization
void for_test_width_1(int *List, int Length) {
#pragma clang loop vectorize_width(1) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_20:.*]]
    List[i] = i * 2;
  }
}

// Verify for loop is not performing vectorization
void for_test_fixed_1(int *List, int Length) {
#pragma clang loop vectorize_width(1, fixed) interleave_count(4) unroll(disable) distribute(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_21:.*]]
    List[i] = i * 2;
  }
}


// Verify unroll attributes are directly attached to the loop metadata
void for_test_vectorize_disable_unroll(int *List, int Length) {
#pragma clang loop vectorize(disable) unroll_count(8)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_22:.*]]
    List[i] = i * 2;
  }
}

// Verify unroll attributes are directly attached to the loop metadata
void for_test_interleave_vectorize_disable_unroll(int *List, int Length) {
#pragma clang loop vectorize(disable) interleave_count(4) unroll_count(8)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_23:.*]]
    List[i] = i * 2;
  }
}

// CHECK-DAG: ![[MP:[0-9]+]] = !{!"llvm.loop.mustprogress"}

// CHECK-DAG: ![[UNROLL_DISABLE:[0-9]+]] = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: ![[UNROLL_8:[0-9]+]] = !{!"llvm.loop.unroll.count", i32 8}
// CHECK-DAG: ![[UNROLL_24:[0-9]+]] = !{!"llvm.loop.unroll.count", i32 24}
// CHECK-DAG: ![[UNROLL_32:[0-9]+]] = !{!"llvm.loop.unroll.count", i32 32}
// CHECK-DAG: ![[UNROLL_FULL:[0-9]+]] = !{!"llvm.loop.unroll.full"}

// CHECK-DAG: ![[DISTRIBUTE_DISABLE:[0-9]+]] = !{!"llvm.loop.distribute.enable", i1 false}

// CHECK-DAG: ![[INTERLEAVE_2:[0-9]+]] = !{!"llvm.loop.interleave.count", i32 2}
// CHECK-DAG: ![[INTERLEAVE_4:[0-9]+]] = !{!"llvm.loop.interleave.count", i32 4}
// CHECK-DAG: ![[INTERLEAVE_8:[0-9]+]] = !{!"llvm.loop.interleave.count", i32 8}
// CHECK-DAG: ![[INTERLEAVE_10:[0-9]+]] = !{!"llvm.loop.interleave.count", i32 10}
// CHECK-DAG: ![[INTERLEAVE_16:[0-9]+]] = !{!"llvm.loop.interleave.count", i32 16}

// CHECK-DAG: ![[VECTORIZE_ENABLE:[0-9]+]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK-DAG: ![[FIXED_VEC:[0-9]+]] = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
// CHECK-DAG: ![[SCALABLE_VEC:[0-9]+]] = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
// CHECK-DAG: ![[WIDTH_1:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 1}
// CHECK-DAG: ![[WIDTH_2:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 2}
// CHECK-DAG: ![[WIDTH_5:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 5}
// CHECK-DAG: ![[WIDTH_6:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 6}
// CHECK-DAG: ![[WIDTH_8:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 8}
// CHECK-DAG: ![[WIDTH_10:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 10}
// CHECK-DAG: ![[WIDTH_16:[0-9]+]] = !{!"llvm.loop.vectorize.width", i32 16}

// CHECK-DAG: ![[ISVECTORIZED:[0-9]+]] = !{!"llvm.loop.isvectorized"}

// CHECK-DAG: ![[LOOP_1]] = distinct !{![[LOOP_1]], ![[MP]], ![[UNROLL_FULL]]}

// CHECK-DAG: ![[LOOP_2]] = distinct !{![[LOOP_2]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_8]], ![[FIXED_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_3]] = distinct !{![[LOOP_3]], ![[MP]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_3:[0-9]+]]}
// CHECK-DAG: ![[FOLLOWUP_VECTOR_3]] = !{!"llvm.loop.vectorize.followup_all", ![[MP]], ![[ISVECTORIZED]], ![[UNROLL_8]]}

// CHECK-DAG: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[WIDTH_2]], ![[FIXED_VEC]], ![[INTERLEAVE_2]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_5]] = distinct !{![[LOOP_5]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_1]]}

// CHECK-DAG: ![[LOOP_6]] = distinct !{![[LOOP_6]], ![[MP]], ![[WIDTH_2]], ![[FIXED_VEC]], ![[INTERLEAVE_2]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_3]]}

// CHECK-DAG: ![[LOOP_7]] = distinct !{![[LOOP_7]], ![[MP]], ![[WIDTH_5]], ![[FIXED_VEC]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_8]] = distinct !{![[LOOP_8]], ![[MP]], ![[WIDTH_5]], ![[FIXED_VEC]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_9]] = distinct !{![[LOOP_9]], ![[MP]], ![[WIDTH_8]], ![[FIXED_VEC]], ![[INTERLEAVE_8]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_3]]}

// CHECK-DAG: ![[LOOP_10]] = distinct !{![[LOOP_10]], ![[MP]], ![[WIDTH_2]], ![[FIXED_VEC]], ![[INTERLEAVE_2]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_3]]}

// CHECK-DAG: ![[LOOP_11]] = distinct !{![[LOOP_11]], ![[MP]], ![[WIDTH_2]], ![[FIXED_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_3]]}

// CHECK-DAG: ![[LOOP_12]] = distinct !{![[LOOP_12]], ![[MP]], ![[WIDTH_6]], ![[FIXED_VEC]], ![[INTERLEAVE_10]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_12:[0-9]+]]}
// CHECK-DAG: ![[FOLLOWUP_VECTOR_12]] = !{!"llvm.loop.vectorize.followup_all", ![[MP]], ![[ISVECTORIZED]], ![[UNROLL_24]]}

// CHECK-DAG: ![[LOOP_13]] = distinct !{![[LOOP_13]], ![[MP]], ![[WIDTH_8]], ![[FIXED_VEC]], ![[INTERLEAVE_16]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_13:[0-9]+]]}
// CHECK-DAG: ![[FOLLOWUP_VECTOR_13]] = !{!"llvm.loop.vectorize.followup_all", ![[MP]], ![[ISVECTORIZED]], ![[UNROLL_32]]}

// CHECK-DAG: ![[LOOP_14]] = distinct !{![[LOOP_14]], ![[MP]], ![[WIDTH_10]], ![[FIXED_VEC]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_15]] = distinct !{![[LOOP_15]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_16]], ![[FIXED_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_16]] = distinct !{![[LOOP_16]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_16]], ![[SCALABLE_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]]}

// CHECK-DAG: ![[LOOP_17]] = distinct !{![[LOOP_17]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[FIXED_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]]}
// CHECK-DAG: ![[LOOP_18]] = distinct !{![[LOOP_18]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[SCALABLE_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]]}
// CHECK-DAG: ![[LOOP_19]] = distinct !{![[LOOP_19]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_1]], ![[SCALABLE_VEC]], ![[INTERLEAVE_4]], ![[VECTORIZE_ENABLE]]}
// CHECK-DAG: ![[LOOP_20]] = distinct !{![[LOOP_20]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_1]], ![[FIXED_VEC]], ![[INTERLEAVE_4]]}
// CHECK-DAG: ![[LOOP_21]] = distinct !{![[LOOP_21]], ![[MP]], ![[UNROLL_DISABLE]], ![[DISTRIBUTE_DISABLE]], ![[WIDTH_1]], ![[FIXED_VEC]], ![[INTERLEAVE_4]]}
// CHECK-DAG: ![[LOOP_22]] = distinct !{![[LOOP_22]], ![[MP]], ![[WIDTH_1]], ![[ISVECTORIZED]], ![[UNROLL_8]]}
// CHECK-DAG: ![[LOOP_23]] = distinct !{![[LOOP_23]], ![[MP]], ![[WIDTH_1]], ![[INTERLEAVE_4]], ![[ISVECTORIZED]], ![[UNROLL_8]]}
