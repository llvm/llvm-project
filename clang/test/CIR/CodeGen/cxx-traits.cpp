// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

namespace std {
struct strong_ordering {
  signed char value;

  constexpr explicit strong_ordering(signed char value) : value(value) {}

  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};

inline constexpr strong_ordering strong_ordering::less(-1);
inline constexpr strong_ordering strong_ordering::equal(0);
inline constexpr strong_ordering strong_ordering::greater(1);
} // namespace std

struct A {};
struct B {};

void expression_trait_expr() {
  bool a = __is_lvalue_expr(0);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[A_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, align 1
// LLVM: store i8 0, ptr %[[A_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 0, ptr %[[A_ADDR]], align 1

void type_trait_expr() {
  enum E {};
  bool a = __is_enum(E);
  bool b = __is_same(int, float);
  bool c = __is_constructible(int, int, int, int);
  bool d = __is_array(int);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[D_ADDR:.*]] = cir.alloca "d" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[CONST_TRUE:.*]] = cir.const #true
// CIR: cir.store {{.*}} %[[CONST_TRUE]], %[[A_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[C_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[D_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, align 1
// LLVM: %[[C_ADDR:.*]] = alloca i8, align 1
// LLVM: %[[D_ADDR:.*]] = alloca i8, align 1
// LLVM: store i8 1, ptr %[[A_ADDR]], align 1
// LLVM: store i8 0, ptr %[[B_ADDR]], align 1
// LLVM: store i8 0, ptr %[[C_ADDR]], align 1
// LLVM: store i8 0, ptr %[[D_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[B_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[C_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[D_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 1, ptr %[[A_ADDR]], align 1
// OGCG: store i8 0, ptr %[[B_ADDR]], align 1
// OGCG: store i8 0, ptr %[[C_ADDR]], align 1
// OGCG: store i8 0, ptr %[[D_ADDR]], align 1

void array_type_trait_expr() {
  unsigned long a = __array_rank(int[10][20]);
  unsigned long b = __array_extent(int[10][20], 1);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!u64i>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} init : !cir.ptr<!u64i>
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !u64i
// CIR: cir.store {{.*}} %[[CONST_2]], %[[A_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_20:.*]] = cir.const #cir.int<20> : !u64i
// CIR: cir.store {{.*}} %[[CONST_20]], %[[B_ADDR]] : !u64i, !cir.ptr<!u64i>

// LLVM: %[[A_ADDR:.*]] = alloca i64, align 8
// LLVM: %[[B_ADDR:.*]] = alloca i64, align 8
// LLVM: store i64 2, ptr %[[A_ADDR]], align 8
// LLVM: store i64 20, ptr %[[B_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[B_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 2, ptr %[[A_ADDR]], align 8
// OGCG: store i64 20, ptr %[[B_ADDR]], align 8

std::strong_ordering strong_ordering_type_trait_equal() {
  return __builtin_type_order(int, int);
}

std::strong_ordering strong_ordering_type_trait_less() {
  return __builtin_type_order(A, B);
}

std::strong_ordering strong_ordering_type_trait_greater() {
  return __builtin_type_order(B, A);
}

// CIR-LABEL: cir.func {{.*}}@_Z32strong_ordering_type_trait_equalv
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !s8i
// CIR: cir.store {{.*}} %[[ZERO]], {{.*}} : !s8i, !cir.ptr<!s8i>

// CIR-LABEL: cir.func {{.*}}@_Z31strong_ordering_type_trait_lessv
// CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s8i
// CIR: cir.store {{.*}} %[[MINUS_ONE]], {{.*}} : !s8i, !cir.ptr<!s8i>

// CIR-LABEL: cir.func {{.*}}@_Z34strong_ordering_type_trait_greaterv
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s8i
// CIR: cir.store {{.*}} %[[ONE]], {{.*}} : !s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}} i8 @_Z32strong_ordering_type_trait_equalv()
// LLVM: store i8 0

// LLVM-LABEL: define{{.*}} i8 @_Z31strong_ordering_type_trait_lessv()
// LLVM: store i8 -1

// LLVM-LABEL: define{{.*}} i8 @_Z34strong_ordering_type_trait_greaterv()
// LLVM: store i8 1

// OGCG-LABEL: define{{.*}} i8 @_Z32strong_ordering_type_trait_equalv()
// OGCG: store i8 0

// OGCG-LABEL: define{{.*}} i8 @_Z31strong_ordering_type_trait_lessv()
// OGCG: store i8 -1

// OGCG-LABEL: define{{.*}} i8 @_Z34strong_ordering_type_trait_greaterv()
// OGCG: store i8 1
