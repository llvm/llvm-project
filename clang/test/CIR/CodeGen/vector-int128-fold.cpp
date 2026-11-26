// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Constant vectors whose elements do not fit in 64 bits are folded at the
// full element width.

typedef __int128 vs2 __attribute__((vector_size(32)));
typedef unsigned __int128 vu2 __attribute__((vector_size(32)));

vs2 result;

void cmp() {
  constexpr vu2 a = {(unsigned __int128)1 << 64, 1};
  constexpr vu2 b = {1, (unsigned __int128)1 << 64};
  result = a < b;
}

// CIR-LABEL: cir.func{{.*}} @_Z3cmpv()
// CIR: cir.const #cir.const_vector<[#cir.int<0> : !s128i, #cir.int<-1> : !s128i]> : !cir.vector<2 x !s128i>

// LLVM-LABEL: define{{.*}} @_Z3cmpv()
// LLVM: store <2 x i128> <i128 0, i128 -1>, ptr @result

void ternary() {
  constexpr vs2 cond = {(__int128)1 << 64, 0};
  constexpr vs2 a = {1, 2};
  constexpr vs2 b = {3, 4};
  result = cond ? a : b;
}

// CIR-LABEL: cir.func{{.*}} @_Z7ternaryv()
// CIR: cir.const #cir.const_vector<[#cir.int<1> : !s128i, #cir.int<4> : !s128i]> : !cir.vector<2 x !s128i>

// LLVM-LABEL: define{{.*}} @_Z7ternaryv()
// LLVM: store <2 x i128> <i128 1, i128 4>, ptr @result

void shuffle_dynamic() {
  constexpr vs2 vec = {1, 2};
  constexpr vu2 indices = {((unsigned __int128)1 << 64) + 1, 0};
  result = __builtin_shufflevector(vec, indices);
}

// CIR-LABEL: cir.func{{.*}} @_Z15shuffle_dynamicv()
// CIR: cir.const #cir.const_vector<[#cir.int<2> : !s128i, #cir.int<1> : !s128i]> : !cir.vector<2 x !s128i>

// LLVM-LABEL: define{{.*}} @_Z15shuffle_dynamicv()
// LLVM: store <2 x i128> <i128 2, i128 1>, ptr @result
