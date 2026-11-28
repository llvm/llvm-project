// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-signed-char -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-signed-char -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-signed-char -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Comparing vectors of char gives a vector of char, which is unsigned here.

typedef char vc4 __attribute__((ext_vector_type(4)));

vc4 result;

void cmp(vc4 a, vc4 b) { result = a < b; }

// CIR-LABEL: cir.func{{.*}} @_Z3cmpDv4_cS_(
// CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !u8i>, !cir.vector<4 x !u8i>

void cmp_constant() {
  constexpr vc4 a = {1, 3, 5, 200};
  constexpr vc4 b = {2, 2, 6, 100};
  result = a < b;
}

// CIR-LABEL: cir.func{{.*}} @_Z12cmp_constantv()
// CIR: cir.const #cir.const_vector<[#cir.int<255> : !u8i, #cir.int<0> : !u8i, #cir.int<255> : !u8i, #cir.int<0> : !u8i]> : !cir.vector<4 x !u8i>

// LLVM-LABEL: define{{.*}} @_Z12cmp_constantv()
// LLVM: store <4 x i8> <i8 -1, i8 0, i8 -1, i8 0>, ptr @result
