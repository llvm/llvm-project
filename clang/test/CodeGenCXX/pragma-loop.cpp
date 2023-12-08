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

// CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], [[MP:![0-9]+]], ![[UNROLL_FULL:.*]]}
// CHECK: ![[UNROLL_FULL]] = !{!"llvm.loop.unroll.full"}

// CHECK: ![[LOOP_2]] = distinct !{![[LOOP_2]], [[MP]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[WIDTH_8:.*]], ![[FIXED_VEC:.*]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE:.*]]}
// CHECK: ![[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
// CHECK: ![[DISTRIBUTE_DISABLE]] = !{!"llvm.loop.distribute.enable", i1 false}
// CHECK: ![[WIDTH_8]] = !{!"llvm.loop.vectorize.width", i32 8}
// CHECK: ![[FIXED_VEC]] = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
// CHECK: ![[INTERLEAVE_4]] = !{!"llvm.loop.interleave.count", i32 4}
// CHECK: ![[VECTORIZE_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}

// CHECK: ![[LOOP_3]] = distinct !{![[LOOP_3]], [[MP]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_3:.*]]}
// CHECK: ![[FOLLOWUP_VECTOR_3]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_3:.*]]}
// CHECK: ![[AFTER_VECTOR_3]] = distinct !{![[AFTER_VECTOR_3]], [[MP]], ![[ISVECTORIZED:.*]], ![[UNROLL_8:.*]]}
// CHECK: ![[ISVECTORIZED]] = !{!"llvm.loop.isvectorized"}
// CHECK: ![[UNROLL_8]] = !{!"llvm.loop.unroll.count", i32 8}

// CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[WIDTH_2:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_2:.*]], ![[VECTORIZE_ENABLE]]}
// CHECK: ![[WIDTH_2]] = !{!"llvm.loop.vectorize.width", i32 2}
// CHECK: ![[INTERLEAVE_2]] = !{!"llvm.loop.interleave.count", i32 2}

// CHECK: ![[LOOP_5]] = distinct !{![[LOOP_5]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[WIDTH_1:.*]]}
// CHECK: ![[WIDTH_1]] = !{!"llvm.loop.vectorize.width", i32 1}

// CHECK: ![[LOOP_6]] = distinct !{![[LOOP_6]], [[MP]], ![[WIDTH_2:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_2:.*]], ![[FOLLOWUP_VECTOR_6:.*]]}
// CHECK: ![[FOLLOWUP_VECTOR_6]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_6:.*]]}
// CHECK: ![[AFTER_VECTOR_6]] = distinct !{![[AFTER_VECTOR_6]], ![[ISVECTORIZED:.*]], ![[UNROLL_8:.*]]}

// CHECK: ![[LOOP_7]] = distinct !{![[LOOP_7]], [[MP]], ![[WIDTH_5:.*]], ![[FIXED_VEC]], ![[VECTORIZE_ENABLE]]}
// CHECK: ![[WIDTH_5]] = !{!"llvm.loop.vectorize.width", i32 5}

// CHECK: ![[LOOP_8]] = distinct !{![[LOOP_8]], [[MP]], ![[WIDTH_5:.*]], ![[FIXED_VEC]], ![[VECTORIZE_ENABLE]]}

// CHECK: ![[LOOP_9]] = distinct !{![[LOOP_9]], ![[WIDTH_8:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_8:.*]], ![[FOLLOWUP_VECTOR_9:.*]]}
// CHECK: ![[FOLLOWUP_VECTOR_9]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_9:.*]]}
// CHECK: ![[AFTER_VECTOR_9]] = distinct !{![[AFTER_VECTOR_9]], ![[ISVECTORIZED:.*]], ![[UNROLL_8:.*]]}

// CHECK: ![[LOOP_10]] = distinct !{![[LOOP_10]], ![[WIDTH_2:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_2:.*]], ![[FOLLOWUP_VECTOR_10:.*]]}
// CHECK: ![[FOLLOWUP_VECTOR_10]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_10:.*]]}
// CHECK: ![[AFTER_VECTOR_10]] = distinct !{![[AFTER_VECTOR_10]], ![[ISVECTORIZED:.*]], ![[UNROLL_8:.*]]}

// CHECK: ![[LOOP_11]] = distinct !{![[LOOP_11]], ![[WIDTH_2:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_4:.*]], ![[FOLLOWUP_VECTOR_11:.*]]}
// CHECK: ![[FOLLOWUP_VECTOR_11]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_11:.*]]}
// CHECK: ![[AFTER_VECTOR_11]] = distinct !{![[AFTER_VECTOR_11]], ![[ISVECTORIZED:.*]], ![[UNROLL_8:.*]]}

// CHECK: ![[LOOP_12]] = distinct !{![[LOOP_12]], ![[WIDTH_6:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_10:.*]], ![[FOLLOWUP_VECTOR_12:.*]]}
// CHECK: ![[FOLLOWUP_VECTOR_12]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_12:.*]]}
// CHECK: ![[AFTER_VECTOR_12]] = distinct !{![[AFTER_VECTOR_12]], ![[ISVECTORIZED:.*]], ![[UNROLL_24:.*]]}
// CHECK: ![[UNROLL_24]] = !{!"llvm.loop.unroll.count", i32 24}

// CHECK: ![[LOOP_13]] = distinct !{![[LOOP_13]], ![[WIDTH_8:.*]], ![[INTERLEAVE_16:.*]], ![[VECTORIZE_ENABLE]], ![[FOLLOWUP_VECTOR_13:.*]]}
// CHECK: ![[INTERLEAVE_16]] = !{!"llvm.loop.interleave.count", i32 16}
// CHECK: ![[FOLLOWUP_VECTOR_13]] = !{!"llvm.loop.vectorize.followup_all", ![[AFTER_VECTOR_13:.*]]}
// CHECK: ![[AFTER_VECTOR_13]] = distinct !{![[AFTER_VECTOR_13]], ![[ISVECTORIZED:.*]], ![[UNROLL_32:.*]]}
// CHECK: ![[UNROLL_32]] = !{!"llvm.loop.unroll.count", i32 32}

// CHECK: ![[LOOP_14]] = distinct !{![[LOOP_14]], [[MP]], ![[WIDTH_10:.*]], ![[FIXED_VEC]], ![[VECTORIZE_ENABLE]]}
// CHECK: ![[WIDTH_10]] = !{!"llvm.loop.vectorize.width", i32 10}

// CHECK: ![[LOOP_15]] = distinct !{![[LOOP_15]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[WIDTH_16:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE:.*]]}
// CHECK: ![[WIDTH_16]] = !{!"llvm.loop.vectorize.width", i32 16}

// CHECK: ![[LOOP_16]] = distinct !{![[LOOP_16]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[WIDTH_16]], ![[SCALABLE_VEC:.*]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE:.*]]}
// CHECK: ![[SCALABLE_VEC]] = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

// CHECK: ![[LOOP_17]] = distinct !{![[LOOP_17]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[FIXED_VEC]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE:.*]]}
// CHECK: ![[LOOP_18]] = distinct !{![[LOOP_18]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[SCALABLE_VEC]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE:.*]]}
// CHECK: ![[LOOP_19]] = distinct !{![[LOOP_19]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]], ![[WIDTH_1]], ![[SCALABLE_VEC]], ![[INTERLEAVE_4:.*]], ![[VECTORIZE_ENABLE:.*]]}
