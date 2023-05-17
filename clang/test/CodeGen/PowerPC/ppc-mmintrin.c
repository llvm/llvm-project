// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P8,CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P8,CHECK,CHECK-LE
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr9 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P9,CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr9 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n| FileCheck %s --check-prefixes=CHECK-P9,CHECK,CHECK-LE

// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P8,CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P8,CHECK,CHECK-LE
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr9 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P9,CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr9 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n| FileCheck %s --check-prefixes=CHECK-P9,CHECK,CHECK-LE

// RUN: %clang -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P8,CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr9 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK-P9,CHECK,CHECK-BE

#include <mmintrin.h>

unsigned long long int ull1, ull2;
int i1, i2;
short s[4];
signed char c[8];
long long int ll1;
__m64 m1, m2, res;

void __attribute__((noinline))
test_add() {
  res = _mm_add_pi32(m1, m2);
  res = _mm_add_pi16(m1, m2);
  res = _mm_add_pi8(m1, m2);
  res = _mm_adds_pu16(m1, m2);
  res = _mm_adds_pu8(m1, m2);
  res = _mm_adds_pi16(m1, m2);
  res = _mm_adds_pi8(m1, m2);
}

// CHECK-LABEL: @test_add

// CHECK-LABEL: define available_externally i64 @_mm_add_pi32
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <4 x i32> @vec_add(int vector[4], int vector[4])
// CHECK-P8: add nsw i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P8: add nsw i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_add_pi16
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <8 x i16> @vec_add(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_add_pi8
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <16 x i8> @vec_add(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally i64 @_mm_adds_pu16
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <8 x i16> @vec_adds(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_adds_pu8
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <16 x i8> @vec_adds(unsigned char vector[16], unsigned char vector[16])

// CHECK-LABEL: define available_externally i64 @_mm_adds_pi16
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <8 x i16> @vec_adds(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_adds_pi8
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <2 x i64> @vec_splats
// CHECK: call <16 x i8> @vec_adds(signed char vector[16], signed char vector[16])

void __attribute__((noinline))
test_alt_name_add() {
  res = _m_paddb(m1, m2);
  res = _m_paddd(m1, m2);
  res = _m_paddsb(m1, m2);
  res = _m_paddsw(m1, m2);
  res = _m_paddusb(m1, m2);
  res = _m_paddusw(m1, m2);
  res = _m_paddw(m1, m2);
}

// CHECK-LABEL: @test_alt_name_add

// CHECK-LABEL: define available_externally i64 @_m_paddb
// CHECK: call i64 @_mm_add_pi8

// CHECK-LABEL: define available_externally i64 @_m_paddd
// CHECK: call i64 @_mm_add_pi32

// CHECK-LABEL: define available_externally i64 @_m_paddsb
// CHECK: call i64 @_mm_adds_pi8

// CHECK-LABEL: define available_externally i64 @_m_paddsw
// CHECK: call i64 @_mm_adds_pi16

// CHECK-LABEL: define available_externally i64 @_m_paddusb
// CHECK: call i64 @_mm_adds_pu8

// CHECK-LABEL: define available_externally i64 @_m_paddusw
// CHECK: call i64 @_mm_adds_pu16

// CHECK-LABEL: define available_externally i64 @_m_paddw
// CHECK: call i64 @_mm_add_pi16

void __attribute__((noinline))
test_cmp() {
  res = _mm_cmpeq_pi32(m1, m2);
  res = _mm_cmpeq_pi16(m1, m2);
  res = _mm_cmpeq_pi8(m1, m2);
  res = _mm_cmpgt_pi32(m1, m2);
  res = _mm_cmpgt_pi16(m1, m2);
  res = _mm_cmpgt_pi8(m1, m2);
}

// CHECK-LABEL: @test_cmp

// CHECK-LABEL: define available_externally i64 @_mm_cmpeq_pi32
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <4 x i32> @vec_cmpeq(int vector[4], int vector[4])
// CHECK-P8: %[[CMP1:[0-9a-zA-Z_.]+]] = icmp eq i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P8: select i1 %[[CMP1]], i32 -1, i32 0
// CHECK-P8: %[[CMP2:[0-9a-zA-Z_.]+]] = icmp eq i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P8: select i1 %[[CMP2]], i32 -1, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_cmpeq_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_cmpeq(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_cmpeq_pi8
// CHECK: call i64 asm "cmpb $0,$1,$2;\0A", "=r,r,r"

// CHECK-LABEL: define available_externally i64 @_mm_cmpgt_pi32
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <4 x i32> @vec_cmpgt(int vector[4], int vector[4])
// CHECK-P8: %[[CMP1:[0-9a-zA-Z_.]+]] = icmp sgt i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P8: select i1 %[[CMP1]], i32 -1, i32 0
// CHECK-P8: [[CMP2:[0-9a-zA-Z_.]+]] = icmp sgt i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P8: select i1 %[[CMP2]], i32 -1, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_cmpgt_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_cmpgt(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_cmpgt_pi8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_cmpgt(signed char vector[16], signed char vector[16])

void __attribute__((noinline))
test_alt_name_cmp() {
  res = _m_pcmpeqb(m1, m2);
  res = _m_pcmpeqd(m1, m2);
  res = _m_pcmpeqw(m1, m2);
  res = _m_pcmpgtb(m1, m2);
  res = _m_pcmpgtd(m1, m2);
  res = _m_pcmpgtw(m1, m2);
}

// CHECK-LABEL: @test_alt_name_cmp

// CHECK-LABEL: define available_externally i64 @_m_pcmpeqb
// CHECK: call i64 @_mm_cmpeq_pi8

// CHECK-LABEL: define available_externally i64 @_m_pcmpeqd
// CHECK: call i64 @_mm_cmpeq_pi32

// CHECK-LABEL: define available_externally i64 @_m_pcmpeqw
// CHECK: call i64 @_mm_cmpeq_pi16

// CHECK-LABEL: define available_externally i64 @_m_pcmpgtb
// CHECK: call i64 @_mm_cmpgt_pi8

// CHECK-LABEL: define available_externally i64 @_m_pcmpgtd
// CHECK: call i64 @_mm_cmpgt_pi32

// CHECK-LABEL: define available_externally i64 @_m_pcmpgtw
// CHECK: call i64 @_mm_cmpgt_pi16

void __attribute__((noinline))
test_convert() {
  ll1 = _mm_cvtm64_si64(m1);
  m1 = _mm_cvtsi32_si64(i1);
  m1 = _mm_cvtsi64_m64(ll1);
  i1 = _mm_cvtsi64_si32(m1);
}

// CHECK-LABEL: @test_convert

// CHECK-LABEL: define available_externally i64 @_mm_cvtm64_si64
// CHECK: %[[RESULT:[0-9a-zA-Z_.]+]] = load i64, ptr %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK-NEXT: ret i64 %[[RESULT]]

// CHECK-LABEL: define available_externally i64 @_mm_cvtsi32_si64
// CHECK: %[[LOAD:[0-9a-zA-Z_.]+]] = load i32, ptr %{{[0-9a-zA-Z_.]+}}
// CHECK-NEXT: %[[RESULT:[0-9a-zA-Z_.]+]] = zext i32 %[[LOAD]] to i64
// CHECK-NEXT: ret i64 %[[RESULT]]

// CHECK-LABEL: define available_externally i64 @_mm_cvtsi64_m64
// CHECK: %[[RESULT:[0-9a-zA-Z_.]+]] = load i64, ptr %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK-NEXT: ret i64 %[[RESULT]]

// CHECK-LABEL: define available_externally signext i32 @_mm_cvtsi64_si32
// CHECK: %[[LOAD:[0-9a-zA-Z_.]+]] = load i64, ptr %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK-NEXT: %[[RESULT:[0-9a-zA-Z_.]+]] = trunc i64 %[[LOAD]] to i32
// CHECK-NEXT: ret i32 %[[RESULT]]

void __attribute__((noinline))
test_alt_name_convert() {
  m1 = _m_from_int(i1);
  m1 = _m_from_int64(ll1);
  i1 = _m_to_int(m1);
  ll1 = _m_to_int64(m1);
}

// CHECK-LABEL: @test_alt_name_convert

// CHECK-LABEL: define available_externally i64 @_m_from_int
// CHECK: call i64 @_mm_cvtsi32_si64

// CHECK-LABEL: define available_externally i64 @_m_from_int64
// CHECK: %[[RESULT:[0-9a-zA-Z_.]+]] = load i64, ptr %{{[0-9a-zA-Z_.]+}}
// CHECK-NEXT: ret i64 %[[RESULT]]

// CHECK-LABEL: define available_externally signext i32 @_m_to_int
// CHECK: call signext i32 @_mm_cvtsi64_si32

// CHECK-LABEL: define available_externally i64 @_m_to_int64
// CHECK: %[[RESULT:[0-9a-zA-Z_.]+]] = load i64, ptr %{{[0-9a-zA-Z_.]+}}
// CHECK-NEXT: ret i64 %[[RESULT]]

void __attribute__((noinline))
test_empty() {
  _mm_empty();
  _m_empty();
}

// CHECK-LABEL: @test_empty

// CHECK-LABEL: define available_externally void @_mm_empty
// CHECK-NEXT: entry
// CHECK-NEXT: ret void

// CHECK-LABEL: define available_externally void @_m_empty
// CHECK-NEXT: entry
// CHECK-NEXT: ret void

void __attribute__((noinline))
test_logic() {
  res = _mm_and_si64(m1, m2);
  res = _mm_andnot_si64(m1, m2);
  res = _mm_or_si64(m1, m2);
  res = _mm_xor_si64(m1, m2);
}

// CHECK-LABEL: @test_logic

// CHECK-LABEL: define available_externally i64 @_mm_and_si64
// CHECK: and i64 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_andnot_si64
// CHECK: %[[XOR:[0-9a-zA-Z_.]+]] = xor i64 %{{[0-9a-zA-Z_.]+}}, -1
// CHECK: and i64 %[[XOR]], %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_or_si64
// CHECK: or i64 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_xor_si64
// CHECK: xor i64 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

void __attribute__((noinline))
test_alt_name_logic() {
  res = _m_pand(m1, m2);
  res = _m_pandn(m1, m2);
  res = _m_por(m1, m2);
  res = _m_pxor(m1, m2);
}

// CHECK-LABEL: @test_alt_name_logic

// CHECK-LABEL: define available_externally i64 @_m_pand
// CHECK: call i64 @_mm_and_si64

// CHECK-LABEL: define available_externally i64 @_m_pandn
// CHECK: call i64 @_mm_andnot_si64

// CHECK-LABEL: define available_externally i64 @_m_por
// CHECK: call i64 @_mm_or_si64

// CHECK-LABEL: define available_externally i64 @_m_pxor
// CHECK: call i64 @_mm_xor_si64

void __attribute__((noinline))
test_madd() {
  res = _mm_madd_pi16(m1, m2);
  res = _m_pmaddwd(m1, m2);
}

// CHECK-LABEL: @test_madd

// CHECK-LABEL: define available_externally i64 @_mm_madd_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <4 x i32> @vec_vmsumshm

// CHECK-LABEL: define available_externally i64 @_m_pmaddwd
// CHECK: call i64 @_mm_madd_pi16

void __attribute__((noinline))
test_mul() {
  res = _mm_mulhi_pi16(m1, m2);
  res = _mm_mullo_pi16(m1, m2);
}

// CHECK-LABEL: @test_mul

// CHECK-LABEL: define available_externally i64 @_mm_mulhi_pi16
// CHECK-BE: store <16 x i8> <i8 0, i8 1, i8 16, i8 17, i8 4, i8 5, i8 20, i8 21, i8 0, i8 1, i8 16, i8 17, i8 4, i8 5, i8 20, i8 21>
// CHECK-LE: store <16 x i8> <i8 2, i8 3, i8 18, i8 19, i8 6, i8 7, i8 22, i8 23, i8 10, i8 11, i8 26, i8 27, i8 14, i8 15, i8 30, i8 31>
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <4 x i32> @vec_vmulesh
// CHECK: call <4 x i32> @vec_vmulosh
// CHECK: call <4 x i32> @vec_perm(int vector[4], int vector[4], unsigned char vector[16])

// CHECK-LABEL: define available_externally i64 @_mm_mullo_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: mul <8 x i16> %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

void __attribute__((noinline))
test_alt_name_mul() {
  res = _m_pmulhw(m1, m2);
  res = _m_pmullw(m1, m2);
}

// CHECK-LABEL: @test_alt_name_mul

// CHECK-LABEL: define available_externally i64 @_m_pmulhw
// CHECK: call i64 @_mm_mulhi_pi16

// CHECK-LABEL: define available_externally i64 @_m_pmullw
// CHECK: call i64 @_mm_mullo_pi16

void __attribute__((noinline))
test_packs() {
  res = _mm_packs_pu16((__m64)ull1, (__m64)ull2);
  res = _mm_packs_pi16((__m64)ull1, (__m64)ull2);
  res = _mm_packs_pi32((__m64)ull1, (__m64)ull2);
}

// CHECK-LABEL: @test_packs

// CHECK-LABEL: define available_externally i64 @_mm_packs_pu16
// CHECK: call <8 x i16> @vec_cmplt
// CHECK: call <16 x i8> @vec_packs(unsigned short vector[8], unsigned short vector[8])
// CHECK: call <16 x i8> @vec_pack(bool vector[8], bool vector[8])
// CHECK: call <16 x i8> @vec_sel(unsigned char vector[16], unsigned char vector[16], bool vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef zeroinitializer, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally i64 @_mm_packs_pi16
// CHECK: call <16 x i8> @vec_packs(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_packs_pi32
// CHECK: call <8 x i16> @vec_packs(int vector[4], int vector[4])

void __attribute__((noinline))
test_alt_name_packs() {
  res = _m_packssdw(m1, m2);
  res = _m_packsswb(m1, m2);
  res = _m_packuswb(m1, m2);
}

// CHECK-LABEL: @test_alt_name_packs

// CHECK-LABEL: define available_externally i64 @_m_packssdw
// CHECK: call i64 @_mm_packs_pi32

// CHECK-LABEL: define available_externally i64 @_m_packsswb
// CHECK: call i64 @_mm_packs_pi16

// CHECK-LABEL: define available_externally i64 @_m_packuswb
// CHECK: call i64 @_mm_packs_pu16

void __attribute__((noinline))
test_set() {
  m1 = _mm_set_pi32(2134, -128);
  m1 = _mm_set_pi16(2134, -128, 1234, 6354);
  m1 = _mm_set_pi8(-128, 10, 0, 123, -1, -5, 127, 5);
}

// CHECK-LABEL: @test_set

// CHECK-LABEL: define available_externally i64 @_mm_set_pi32
// CHECK-COUNT-2: store i32 %{{[0-9a-zA-Z_.]+}}, ptr

// CHECK-LABEL: define available_externally i64 @_mm_set_pi16
// CHECK-COUNT-4: store i16 %{{[0-9a-zA-Z_.]+}}, ptr

// CHECK-LABEL: define available_externally i64 @_mm_set_pi8
// CHECK-COUNT-8: store i8 %{{[0-9a-zA-Z_.]+}}, ptr

void __attribute__((noinline))
test_set1() {
  res = _mm_set1_pi32(i1);
  res = _mm_set1_pi16(s[0]);
  res = _mm_set1_pi8(c[0]);
}

// CHECK-LABEL: @test_set1

// CHECK-LABEL: define available_externally i64 @_mm_set1_pi32
// CHECK-COUNT-2: store i32 %{{[0-9a-zA-Z_.]+}}, ptr

// CHECK-LABEL: define available_externally i64 @_mm_set1_pi16
// CHECK-P9: call <8 x i16> @vec_splats(short)
// CHECK-P9: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK-P8: %[[ADDR1:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK-P8: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR1]], align 8
// CHECK-P8: %[[ADDR2:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK-P8: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR2]], align 2
// CHECK-P8: %[[ADDR3:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK-P8: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR3]], align 4
// CHECK-P8: %[[ADDR4:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK-P8: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR4]], align 2


// CHECK-LABEL: define available_externally i64 @_mm_set1_pi8
// CHECK: call <16 x i8> @vec_splats(signed char)
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <16 x i8> %{{[0-9a-zA-Z_.]+}} to <2 x i64>
// CHECK: extractelement <2 x i64> %[[CAST]], i32 0

void __attribute__((noinline))
test_setr() {
  res = _mm_setr_pi32(i1, i2);
  res = _mm_setr_pi16(s[0], s[1], s[2], s[3]);
  res = _mm_setr_pi8(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

// CHECK-LABEL: @test_setr

// CHECK-LABEL: define available_externally i64 @_mm_setr_pi32
// CHECK: %[[ADDR1:[0-9a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: store i32 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR1]], align 8
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: store i32 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR2]], align 4

// CHECK-LABEL: define available_externally i64 @_mm_setr_pi16
// CHECK: call i64 @_mm_set_pi16

// CHECK-LABEL: define available_externally i64 @_mm_setr_pi8
// CHECK: call i64 @_mm_set_pi8

void __attribute__((noinline))
test_setzero() {
  res = _mm_setzero_si64();
}

// CHECK-LABEL: @test_setzero

// CHECK-LABEL: define available_externally i64 @_mm_setzero_si64
// CHECK: entry
// CHECK-NEXT: ret i64 0

void __attribute__((noinline))
test_sll() {
  res = _mm_sll_pi16(m1, m2);
  res = _mm_sll_pi32(m1, m2);
  res = _mm_sll_si64(m1, m2);
  res = _mm_slli_pi16(m1, i1);
  res = _mm_slli_pi32(m1, i1);
  res = _mm_slli_si64(m1, i1);
}

// CHECK-LABEL: @test_sll

// CHECK-LABEL: define available_externally i64 @_mm_sll_pi16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp ule i64 %{{[0-9a-zA-Z_.]+}}, 15
// CHECK-NEXT: br i1 %[[CMP]]
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: trunc i64 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: call <8 x i16> @vec_splats(unsigned short)
// CHECK: call <8 x i16> @vec_sl(short vector[8], unsigned short vector[8])
// CHECK: store i64 0, ptr

// CHECK-LABEL: define available_externally i64 @_mm_sll_pi32
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: shl i32 %{{[0-9a-zA-Z_.]+}}, %[[TRUNC]]
// CHECK: trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: shl i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_sll_si64
// CHECK: shl i64 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_slli_pi16
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call i64 @_mm_sll_pi16(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %[[EXT]])

// CHECK-LABEL: define available_externally i64 @_mm_slli_pi32
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call i64 @_mm_sll_pi32(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %[[EXT]])

// CHECK-LABEL: define available_externally i64 @_mm_slli_si64
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = zext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: shl i64 %{{[0-9a-zA-Z_.]+}}, %[[EXT]]

void __attribute__((noinline))
test_alt_name_sll() {
  res = _m_pslld(m1, m2);
  res = _m_pslldi(m1, i1);
  res = _m_psllq(m1, m2);
  res = _m_psllqi(m1, i1);
  res = _m_psllw(m1, m2);
  res = _m_psllwi(m1, i1);
}

// CHECK-LABEL: @test_alt_name_sll

// CHECK-LABEL: define available_externally i64 @_m_pslld
// CHECK: call i64 @_mm_sll_pi32

// CHECK-LABEL: define available_externally i64 @_m_pslldi
// CHECK: call i64 @_mm_slli_pi32

// CHECK-LABEL: define available_externally i64 @_m_psllq
// CHECK: call i64 @_mm_sll_si64

// CHECK-LABEL: define available_externally i64 @_m_psllqi
// CHECK: call i64 @_mm_slli_si64

// CHECK-LABEL: define available_externally i64 @_m_psllw
// CHECK: call i64 @_mm_sll_pi16

// CHECK-LABEL: define available_externally i64 @_m_psllwi
// CHECK: call i64 @_mm_slli_pi16

void __attribute__((noinline))
test_sra() {
  res = _mm_sra_pi32(m1, m2);
  res = _mm_sra_pi16(m1, m2);
  res = _mm_srai_pi32(m1, i1);
  res = _mm_srai_pi16(m1, i1);
}

// CHECK-LABEL: @test_sra

// CHECK-LABEL: define available_externally i64 @_mm_sra_pi32
// CHECK: %[[TRUNC1:[0-9a-zA-Z_.]+]] = trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: ashr i32 %{{[0-9a-zA-Z_.]+}}, %[[TRUNC1]]
// CHECK: %[[TRUNC2:[0-9a-zA-Z_.]+]] = trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: ashr i32 %{{[0-9a-zA-Z_.]+}}, %[[TRUNC2]]

// CHECK-LABEL: define available_externally i64 @_mm_sra_pi16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp ule i64 %{{[0-9a-zA-Z_.]+}}, 15
// CHECK: br i1 %[[CMP]]
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i64 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: call <8 x i16> @vec_splats(unsigned short)(i16 noundef zeroext %[[TRUNC]])
// CHECK: call <8 x i16> @vec_sra(short vector[8], unsigned short vector[8])
// CHECK: store i64 0, ptr

// CHECK-LABEL: define available_externally i64 @_mm_srai_pi32
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call i64 @_mm_sra_pi32(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %[[EXT]])

// CHECK-LABEL: define available_externally i64 @_mm_srai_pi16
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call i64 @_mm_sra_pi16(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %[[EXT]])

void __attribute__((noinline))
test_alt_name_sra() {
  res = _m_psrad(m1, m2);
  res = _m_psraw(m1, m2);
  res = _m_psradi(m1, i1);
  res = _m_psrawi(m1, i1);
}

// CHECK-LABEL: @test_alt_name_sra

// CHECK-LABEL: define available_externally i64 @_m_psrad
// CHECK: call i64 @_mm_sra_pi32

// CHECK-LABEL: define available_externally i64 @_m_psraw
// CHECK: call i64 @_mm_sra_pi16

// CHECK-LABEL: define available_externally i64 @_m_psradi
// CHECK: call i64 @_mm_srai_pi32

// CHECK-LABEL: define available_externally i64 @_m_psrawi
// CHECK: call i64 @_mm_srai_pi16

void __attribute__((noinline))
test_srl() {
  res = _mm_srl_si64(m1, m2);
  res = _mm_srl_pi32(m1, m2);
  res = _mm_srl_pi16(m1, m2);
  res = _mm_srli_si64(m1, i1);
  res = _mm_srli_pi32(m1, i1);
  res = _mm_srli_pi16(m1, i1);
}

// CHECK-LABEL: @test_srl

// CHECK-LABEL: define available_externally i64 @_mm_srl_si64
// CHECK: lshr i64 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}

// CHECK-LABEL: define available_externally i64 @_mm_srl_pi32
// CHECK: %[[TRUNC1:[0-9a-zA-Z_.]+]] = trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: lshr i32 %{{[0-9a-zA-Z_.]+}}, %[[TRUNC1]]
// CHECK: %[[TRUNC2:[0-9a-zA-Z_.]+]] = trunc i64 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: lshr i32 %{{[0-9a-zA-Z_.]+}}, %[[TRUNC2]]

// CHECK-LABEL: define available_externally i64 @_mm_srl_pi16
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = icmp ule i64 %{{[0-9a-zA-Z_.]+}}, 15
// CHECK: br i1 %[[CMP]]
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: trunc i64 %{{[0-9a-zA-Z_.]+}} to i16
// CHECK: call <8 x i16> @vec_splats(unsigned short)
// CHECK: call <8 x i16> @vec_sr(unsigned short vector[8], unsigned short vector[8])
// CHECK: store i64 0, ptr %{{[0-9a-zA-Z_.]+}}, align 8

// CHECK-LABEL: define available_externally i64 @_mm_srli_si64
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = zext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: lshr i64 %{{[0-9a-zA-Z_.]+}}, %[[EXT]]

// CHECK-LABEL: define available_externally i64 @_mm_srli_pi32
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call i64 @_mm_srl_pi32(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %[[EXT]])

// CHECK-LABEL: define available_externally i64 @_mm_srli_pi16
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: call i64 @_mm_srl_pi16(i64 noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %[[EXT]])

void __attribute__((noinline))
test_alt_name_srl() {
  res = _m_psrlq(m1, m2);
  res = _m_psrld(m1, m2);
  res = _m_psrlw(m1, m2);
  res = _m_psrlqi(m1, i1);
  res = _m_psrldi(m1, i1);
  res = _m_psrlwi(m1, i1);
}

// CHECK-LABEL: @test_alt_name_srl

// CHECK-LABEL: define available_externally i64 @_m_psrlq
// CHECK: call i64 @_mm_srl_si64

// CHECK-LABEL: define available_externally i64 @_m_psrld
// CHECK: call i64 @_mm_srl_pi32

// CHECK-LABEL: define available_externally i64 @_m_psrlw
// CHECK: call i64 @_mm_srl_pi16

// CHECK-LABEL: define available_externally i64 @_m_psrlqi
// CHECK: call i64 @_mm_srli_si64

// CHECK-LABEL: define available_externally i64 @_m_psrldi
// CHECK: call i64 @_mm_srli_pi32

// CHECK-LABEL: define available_externally i64 @_m_psrlwi
// CHECK: call i64 @_mm_srli_pi16

void __attribute__((noinline))
test_sub() {
  res = _mm_sub_pi32(m1, m2);
  res = _mm_sub_pi16(m1, m2);
  res = _mm_sub_pi8(m1, m2);
  res = _mm_subs_pi16(m1, m2);
  res = _mm_subs_pi8(m1, m2);
  res = _mm_subs_pu16(m1, m2);
  res = _mm_subs_pu8(m1, m2);
}

// CHECK-LABEL: @test_sub

// CHECK-LABEL: define available_externally i64 @_mm_sub_pi32
// CHECK-P8-COUNT-2: getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK-P8: sub nsw i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P8-COUNT-2: getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK-P8: sub nsw i32 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <2 x i64> @vec_splats(unsigned long long)
// CHECK-P9: call <4 x i32> @vec_sub(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally i64 @_mm_sub_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_sub(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_sub_pi8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_sub(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally i64 @_mm_subs_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_subs(short vector[8], short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_subs_pi8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_subs(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally i64 @_mm_subs_pu16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_subs(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally i64 @_mm_subs_pu8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_subs(unsigned char vector[16], unsigned char vector[16])

void __attribute__((noinline))
test_alt_name_sub() {
  res = _m_psubd(m1, m2);
  res = _m_psubw(m1, m2);
  res = _m_psubb(m1, m2);
  res = _m_psubsw(m1, m2);
  res = _m_psubsb(m1, m2);
  res = _m_psubusw(m1, m2);
  res = _m_psubusb(m1, m2);
}

// CHECK-LABEL: @test_alt_name_sub

// CHECK-LABEL: define available_externally i64 @_m_psubd
// CHECK: call i64 @_mm_sub_pi32

// CHECK-LABEL: define available_externally i64 @_m_psubw
// CHECK: call i64 @_mm_sub_pi16

// CHECK-LABEL: define available_externally i64 @_m_psubb
// CHECK: call i64 @_mm_sub_pi8

// CHECK-LABEL: define available_externally i64 @_m_psubsw
// CHECK: call i64 @_mm_subs_pi16

// CHECK-LABEL: define available_externally i64 @_m_psubsb
// CHECK: call i64 @_mm_subs_pi8

// CHECK-LABEL: define available_externally i64 @_m_psubusw
// CHECK: call i64 @_mm_subs_pu16

// CHECK-LABEL: define available_externally i64 @_m_psubusb
// CHECK: call i64 @_mm_subs_pu8

void __attribute__((noinline))
test_unpack() {
  res = _mm_unpackhi_pi32(m1, m2);
  res = _mm_unpackhi_pi16(m1, m2);
  res = _mm_unpackhi_pi8(m1, m2);
  res = _mm_unpacklo_pi32(m1, m2);
  res = _mm_unpacklo_pi16(m1, m2);
  res = _mm_unpacklo_pi8(m1, m2);
}

// CHECK-LABEL: @test_unpack

// CHECK-LABEL: define available_externally i64 @_mm_unpackhi_pi32
// CHECK: getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: %[[ADDR1:[0-9a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: store i32 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR1]], align 8
// CHECK: getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: store i32 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR2]], align 4

// CHECK-LABEL: define available_externally i64 @_mm_unpackhi_pi16
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK: %[[ADDR1:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR1]], align 8
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR2]], align 2
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK: %[[ADDR3:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR3]], align 4
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK: %[[ADDR4:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR4]], align 2

// CHECK-LABEL: define available_externally i64 @_mm_unpackhi_pi8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_mergel(unsigned char vector[16], unsigned char vector[16])

// CHECK-LABEL: define available_externally i64 @_mm_unpacklo_pi32
// CHECK: getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: %[[ADDR1:[0-9a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: store i32 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR1]], align 8
// CHECK: getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: store i32 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR2]], align 4

// CHECK-LABEL: define available_externally i64 @_mm_unpacklo_pi16
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: %[[ADDR1:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR1]], align 8
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: %[[ADDR2:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR2]], align 2
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: %[[ADDR3:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR3]], align 4
// CHECK: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: %[[ADDR4:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK: store i16 %{{[0-9a-zA-Z_.]+}}, ptr %[[ADDR4]], align 2

// CHECK-LABEL: define available_externally i64 @_mm_unpacklo_pi8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_mergel(unsigned char vector[16], unsigned char vector[16])

void __attribute__((noinline))
test_alt_name_unpack() {
  res = _m_punpckhdq(m1, m2);
  res = _m_punpckhwd(m1, m2);
  res = _m_punpckhbw(m1, m2);
  res = _m_punpckldq(m1, m2);
  res = _m_punpcklwd(m1, m2);
  res = _m_punpcklbw(m1, m2);
}

// CHECK-LABEL: @test_alt_name_unpack

// CHECK-LABEL: define available_externally i64 @_m_punpckhdq
// CHECK: call i64 @_mm_unpackhi_pi32

// CHECK-LABEL: define available_externally i64 @_m_punpckhwd
// CHECK: call i64 @_mm_unpackhi_pi16

// CHECK-LABEL: define available_externally i64 @_m_punpckhbw
// CHECK: call i64 @_mm_unpackhi_pi8

// CHECK-LABEL: define available_externally i64 @_m_punpckldq
// CHECK: call i64 @_mm_unpacklo_pi32

// CHECK-LABEL: define available_externally i64 @_m_punpcklwd
// CHECK: call i64 @_mm_unpacklo_pi16

// CHECK-LABEL: define available_externally i64 @_m_punpcklbw
// CHECK: call i64 @_mm_unpacklo_pi8
