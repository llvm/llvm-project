// RUN: %clang_cc1 -no-enable-noundef-analysis -triple riscv64 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,RV64
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple riscv32 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,RV32
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple riscv64 -target-feature +d -target-abi lp64d -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,RV64

// REQUIRES: riscv-registered-target

#define SWIFTCALL __attribute__((swiftcall))
#define OUT __attribute__((swift_indirect_result))
#define ERROR __attribute__((swift_error_result))
#define CONTEXT __attribute__((swift_context))

/*****************************************************************************/
/****************************** PARAMETER ABIS *******************************/
/*****************************************************************************/

SWIFTCALL void indirect_result_1(OUT int *arg0, OUT float *arg1) {}
// CHECK-LABEL: define {{.*}} void @indirect_result_1(ptr noalias sret(ptr) align 4 dereferenceable(4){{.*}}, ptr noalias align 4 dereferenceable(4){{.*}})

SWIFTCALL void context_1(CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_1(ptr swiftself

SWIFTCALL void context_2(void *arg0, CONTEXT void *self) {}
// CHECK-LABEL: define {{.*}} void @context_2(ptr{{.*}}, ptr swiftself

SWIFTCALL void context_error_1(CONTEXT int *self, ERROR float **error) {}
// CHECK-LABEL: define {{.*}} void @context_error_1(ptr swiftself{{.*}}, ptr swifterror %0)

/*****************************************************************************/
/********************************** LOWERING *********************************/
/*****************************************************************************/

#define TEST(TYPE)                       \
  SWIFTCALL TYPE return_##TYPE(void) {   \
    TYPE result = {};                    \
    return result;                       \
  }                                      \
  SWIFTCALL void take_##TYPE(TYPE v) {   \
  }                                      \
  void test_##TYPE(void) {               \
    take_##TYPE(return_##TYPE());        \
  }

// Sub-pointer-sized fields merge into XLen chunks.
typedef struct {
  int x;
  int y;
} struct_2ints;
TEST(struct_2ints)
// RV64-LABEL: define {{.*}} swiftcc i64 @return_struct_2ints()
// RV32-LABEL: define {{.*}} swiftcc { i32, i32 } @return_struct_2ints()
// RV64-LABEL: define {{.*}} swiftcc void @take_struct_2ints(i64
// RV32-LABEL: define {{.*}} swiftcc void @take_struct_2ints(i32 %0, i32 %1)

// Four XLen-sized components are returned directly; the backend returns
// them in a0-a3. On riscv32 the same struct occupies eight registers and is
// returned indirectly.
typedef struct {
  long long a, b, c, d;
} struct_4i64;
TEST(struct_4i64)
// RV64-LABEL: define {{.*}} swiftcc { i64, i64, i64, i64 } @return_struct_4i64()
// RV32-LABEL: define {{.*}} swiftcc void @return_struct_4i64(ptr dead_on_unwind noalias writable sret
// RV64-LABEL: define {{.*}} swiftcc void @take_struct_4i64(i64 %0, i64 %1, i64 %2, i64 %3)
// RV32-LABEL: define {{.*}} swiftcc void @take_struct_4i64(ptr

// Five components exceed the four-register return budget everywhere.
typedef struct {
  long long a, b, c, d, e;
} struct_5i64;
TEST(struct_5i64)
// CHECK-LABEL: define {{.*}} swiftcc void @return_struct_5i64(ptr dead_on_unwind noalias writable sret
// RV64-LABEL: define {{.*}} swiftcc void @take_struct_5i64(ptr
// RV32-LABEL: define {{.*}} swiftcc void @take_struct_5i64(ptr

// The mixed case from the swiftcall review: three components, returned
// directly on riscv64 (a0/fa0/a1 under a hard-float ABI).
typedef struct {
  long long a;
  float f;
  long long b;
} struct_mixed;
TEST(struct_mixed)
// RV64-LABEL: define {{.*}} swiftcc { i64, float, i64 } @return_struct_mixed()
// RV32-LABEL: define {{.*}} swiftcc void @return_struct_mixed(ptr dead_on_unwind noalias writable sret
// RV64-LABEL: define {{.*}} swiftcc void @take_struct_mixed(i64 %0, float %1, i64 %2)

// Vector types are lowered into scalar components: the base RISC-V calling
// convention has no vector registers.
typedef float float4 __attribute__((ext_vector_type(4)));
TEST(float4)
// RV64-LABEL: define {{.*}} swiftcc { float, float, float, float } @return_float4()
// RV32-LABEL: define {{.*}} swiftcc { float, float, float, float } @return_float4()
// RV64-LABEL: define {{.*}} swiftcc void @take_float4(float %0, float %1, float %2, float %3)
