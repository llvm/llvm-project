// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm  -target-cpu skylake-avx512 < %s | FileCheck %s --check-prefixes=CHECK,AVX
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm  -target-cpu skylake-avx512 < %s | FileCheck %s --check-prefixes=CHECK,AVX
// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm < %s | FileCheck %s --check-prefixes=CHECK,NOAVX
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm < %s | FileCheck %s --check-prefixes=CHECK,NOAVX

#define SYSV_CC __attribute__((sysv_abi))

// Make sure we coerce structs according to the SysV rules instead of passing
// them indirectly as we would for Win64.
struct StringRef {
  char *Str;
  __SIZE_TYPE__ Size;
};
extern volatile char gc;
void SYSV_CC take_stringref(struct StringRef s);
void callit(void) {
  struct StringRef s = {"asdf", 4};
  take_stringref(s);
}
// CHECK: define {{(dso_local )?}}void @callit()
// CHECK: call {{(x86_64_sysvcc )?}}void @take_stringref(i8* {{[^,]*}}, i64 {{[^,]*}})
// CHECK: declare {{(dso_local )?}}{{(x86_64_sysvcc )?}}void @take_stringref(i8*, i64)

// Check that we pass vectors directly if the target feature is enabled, and
// not otherwise.
typedef __attribute__((vector_size(32))) float my_m256;
typedef __attribute__((vector_size(64))) float my_m512;

my_m256 SYSV_CC get_m256(void);
void SYSV_CC take_m256(my_m256);
my_m512 SYSV_CC get_m512(void);
void SYSV_CC take_m512(my_m512);

void use_vectors(void) {
  my_m256 v1 = get_m256();
  take_m256(v1);
  my_m512 v2 = get_m512();
  take_m512(v2);
}

// CHECK: define {{(dso_local )?}}void @use_vectors()
// AVX: call {{(x86_64_sysvcc )?}}<8 x float> @get_m256()
// AVX: call {{(x86_64_sysvcc )?}}void @take_m256(<8 x float> noundef %{{.*}})
// AVX: call {{(x86_64_sysvcc )?}}<16 x float> @get_m512()
// AVX: call {{(x86_64_sysvcc )?}}void @take_m512(<16 x float> noundef %{{.*}})
// NOAVX: call {{(x86_64_sysvcc )?}}<8 x float> @get_m256()
// NOAVX: call {{(x86_64_sysvcc )?}}void @take_m256(<8 x float>* noundef byval(<8 x float>) align 32 %{{.*}})
// NOAVX: call {{(x86_64_sysvcc )?}}<16 x float> @get_m512()
// NOAVX: call {{(x86_64_sysvcc )?}}void @take_m512(<16 x float>* noundef byval(<16 x float>) align 64 %{{.*}})
