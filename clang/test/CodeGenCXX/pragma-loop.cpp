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

// CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], ![[WIDTH_4:.*]], ![[INTERLEAVE_4:.*]], ![[INTENABLE_1:.*]], ![[UNROLL_FULL:.*]], ![[DISTRIBUTE_ENABLE:.*]]}
// CHECK: ![[WIDTH_4]] = !{!"llvm.loop.vectorize.width", i32 4}
// CHECK: ![[INTERLEAVE_4]] = !{!"llvm.loop.interleave.count", i32 4}
// CHECK: ![[INTENABLE_1]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: ![[UNROLL_FULL]] = !{!"llvm.loop.unroll.full"}
// CHECK: ![[DISTRIBUTE_ENABLE]] = !{!"llvm.loop.distribute.enable", i1 true}
// CHECK: ![[LOOP_2]] = distinct !{![[LOOP_2:.*]], ![[WIDTH_8:.*]], ![[INTERLEAVE_4:.*]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]]}
// CHECK: ![[WIDTH_8]] = !{!"llvm.loop.vectorize.width", i32 8}
// CHECK: ![[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
// CHECK: ![[DISTRIBUTE_DISABLE]] = !{!"llvm.loop.distribute.enable", i1 false}
// CHECK: ![[LOOP_3]] = distinct !{![[LOOP_3]], ![[INTERLEAVE_4:.*]], ![[UNROLL_8:.*]], ![[INTENABLE_1:.*]]}
// CHECK: ![[UNROLL_8]] = !{!"llvm.loop.unroll.count", i32 8}
// CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[WIDTH_2:.*]], ![[INTERLEAVE_2:.*]]}
// CHECK: ![[WIDTH_2]] = !{!"llvm.loop.vectorize.width", i32 2}
// CHECK: ![[INTERLEAVE_2]] = !{!"llvm.loop.interleave.count", i32 2}
// CHECK: ![[LOOP_5]] = distinct !{![[LOOP_5]], ![[WIDTH_1:.*]], ![[UNROLL_DISABLE:.*]], ![[DISTRIBUTE_DISABLE:.*]]}
// CHECK: ![[WIDTH_1]] = !{!"llvm.loop.vectorize.width", i32 1}
// CHECK: ![[LOOP_6]] = distinct !{![[LOOP_6]], ![[WIDTH_2:.*]], ![[INTERLEAVE_2:.*]], ![[UNROLL_8:.*]]}
// CHECK: ![[LOOP_7]] = distinct !{![[LOOP_7]], ![[WIDTH_5:.*]]}
// CHECK: ![[WIDTH_5]] = !{!"llvm.loop.vectorize.width", i32 5}
// CHECK: ![[LOOP_8]] = distinct !{![[LOOP_8]], ![[WIDTH_5:.*]]}
// CHECK: ![[LOOP_9]] = distinct !{![[LOOP_9]], ![[WIDTH_8:.*]], ![[INTERLEAVE_8:.*]], ![[UNROLL_8:.*]]}
// CHECK: ![[INTERLEAVE_8]] = !{!"llvm.loop.interleave.count", i32 8}
// CHECK: ![[LOOP_10]] = distinct !{![[LOOP_10]], ![[WIDTH_2:.*]], ![[INTERLEAVE_2:.*]], ![[UNROLL_8:.*]]}
// CHECK: ![[LOOP_11]] = distinct !{![[LOOP_11]], ![[WIDTH_2:.*]], ![[INTERLEAVE_4:.*]], ![[UNROLL_8:.*]]}
// CHECK: ![[LOOP_12]] = distinct !{![[LOOP_12]], ![[WIDTH_6:.*]], ![[INTERLEAVE_10:.*]], ![[UNROLL_24:.*]]}
// CHECK: ![[WIDTH_6]] = !{!"llvm.loop.vectorize.width", i32 6}
// CHECK: ![[INTERLEAVE_10]] = !{!"llvm.loop.interleave.count", i32 10}
// CHECK: ![[UNROLL_24]] = !{!"llvm.loop.unroll.count", i32 24}
// CHECK: ![[LOOP_13]] = distinct !{![[LOOP_13]], ![[WIDTH_8:.*]], ![[INTERLEAVE_16:.*]], ![[UNROLL_32:.*]]}
// CHECK: ![[INTERLEAVE_16]] = !{!"llvm.loop.interleave.count", i32 16}
// CHECK: ![[UNROLL_32]] = !{!"llvm.loop.unroll.count", i32 32}
// CHECK: ![[LOOP_14]] = distinct !{![[LOOP_14]], ![[WIDTH_10:.*]]}
// CHECK: ![[WIDTH_10]] = !{!"llvm.loop.vectorize.width", i32 10}
