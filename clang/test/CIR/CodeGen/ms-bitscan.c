// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions \
// RUN:   -Wno-implicit-function-declaration -fclangir \
// RUN:   -triple x86_64-unknown-linux -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

#ifdef __LP64__
#define LONG int
#else
#define LONG long
#endif

#ifdef __cplusplus
extern "C" {
#endif

unsigned char test_BitScanForward(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanForward(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanForward
// CIR: [[ZERO_FWD:%.*]] = cir.const #cir.int<0> : !u32i
// CIR: [[NONZERO_FWD:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_FWD]] : !u32i
// CIR: cir.if [[NONZERO_FWD]] {
// CIR: cir.ctz {{%.*}} poison_zero : !u32i
// CIR: cir.store align(4) {{%.*}}, {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR: cir.cast bool_to_int [[NONZERO_FWD]] : !cir.bool -> !u8i

unsigned char test_BitScanReverse(unsigned LONG *Index, unsigned LONG Mask) {
  return _BitScanReverse(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanReverse
// CIR: [[ZERO_REV:%.*]] = cir.const #cir.int<0> : !u32i
// CIR: [[NONZERO_REV:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_REV]] : !u32i
// CIR: cir.if [[NONZERO_REV]] {
// CIR: cir.const #cir.int<31> : !u32i
// CIR: cir.clz {{%.*}} poison_zero : !u32i
// CIR: cir.sub nsw {{%.*}}, {{%.*}} : !u32i
// CIR: cir.store align(4) {{%.*}}, {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR: cir.cast bool_to_int [[NONZERO_REV]] : !cir.bool -> !u8i

#if defined(__x86_64__)
unsigned char test_BitScanForward64(unsigned LONG *Index,
                                    unsigned __int64 Mask) {
  return _BitScanForward64(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanForward64
// CIR: [[ZERO_FWD64:%.*]] = cir.const #cir.int<0> : !u64i
// CIR: [[NONZERO_FWD64:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_FWD64]] : !u64i
// CIR: cir.if [[NONZERO_FWD64]] {
// CIR: cir.ctz {{%.*}} poison_zero : !u64i
// CIR: cir.cast integral {{%.*}} : !u64i -> !u32i
// CIR: cir.store align(4) {{%.*}}, {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR: cir.cast bool_to_int [[NONZERO_FWD64]] : !cir.bool -> !u8i

unsigned char test_BitScanReverse64(unsigned LONG *Index,
                                    unsigned __int64 Mask) {
  return _BitScanReverse64(Index, Mask);
}

// CIR-LABEL: cir.func{{.*}}@test_BitScanReverse64
// CIR: [[ZERO_REV64:%.*]] = cir.const #cir.int<0> : !u64i
// CIR: [[NONZERO_REV64:%.*]] = cir.cmp ne {{%.*}}, [[ZERO_REV64]] : !u64i
// CIR: cir.if [[NONZERO_REV64]] {
// CIR: cir.const #cir.int<63> : !u32i
// CIR: cir.clz {{%.*}} poison_zero : !u64i
// CIR: cir.cast integral {{%.*}} : !u64i -> !u32i
// CIR: cir.sub nsw {{%.*}}, {{%.*}} : !u32i
// CIR: cir.store align(4) {{%.*}}, {{%.*}} : !u32i, !cir.ptr<!u32i>
// CIR: cir.cast bool_to_int [[NONZERO_REV64]] : !cir.bool -> !u8i
#endif

#ifdef __cplusplus
}
#endif