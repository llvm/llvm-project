// RUN: %clang_cc1 -ffreestanding %s -triple=i386-apple-darwin -target-feature +shstk -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +shstk  -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=X86_64

#include <immintrin.h>

void test_incsspd(int a) {
  // CHECK-LABEL: @test_incsspd
  // CHECK:       call void @llvm.x86.incsspd(i32 %{{[0-9]+}})
  _incsspd(a);
}

#ifdef __x86_64__
void test_incsspq(int a) {
  // X86_64-LABEL: @test_incsspq
  // X86_64:       call void @llvm.x86.incsspq(i64 %{{[a-z0-9.]+}})
  _incsspq(a);
}
#endif

unsigned int test_rdsspd(unsigned int a) {
  // CHECK-LABEL: @test_rdsspd
  // CHECK:       call i32 @llvm.x86.rdsspd(i32 %{{[a-z0-9.]+}})
  return _rdsspd(a);
}

#ifdef __x86_64__
unsigned long long test_rdsspq(unsigned long long a) {
  // X86_64-LABEL: @test_rdsspq
  // X86_64:       call i64 @llvm.x86.rdsspq(i64 %{{[a-z0-9.]+}})
  return _rdsspq(a);
}
#endif

void  test_saveprevssp() {
  // CHECK-LABEL: @test_saveprevssp
  // CHECK:       call void @llvm.x86.saveprevssp()
  _saveprevssp();
}

void test_rstorssp(void * __p) {
  // CHECK-LABEL: @test_rstorssp
  // CHECK:       call void @llvm.x86.rstorssp(i8* %{{[a-z0-9.]+}})
  _rstorssp(__p);
}

void test_wrssd(unsigned int __a, void * __p) {
  // CHECK-LABEL: @test_wrssd
  // CHECK:       call void @llvm.x86.wrssd(i32 %{{[a-z0-9.]+}}, i8* %{{[a-z0-9.]+}})
  _wrssd(__a, __p);
}

#ifdef __x86_64__
void test_wrssq(unsigned long long __a, void * __p) {
  // X86_64-LABEL: @test_wrssq
  // X86_64:       call void @llvm.x86.wrssq(i64 %{{[a-z0-9.]+}}, i8* %{{[a-z0-9.]+}})
  _wrssq(__a, __p);
}
#endif

void test_wrussd(unsigned int __a, void * __p) {
  // CHECK-LABEL: @test_wrussd
  // CHECK:       call void @llvm.x86.wrussd(i32 %{{[a-z0-9.]+}}, i8* %{{[a-z0-9.]+}})
  _wrussd(__a, __p);
}

#ifdef __x86_64__
void test_wrussq(unsigned long long __a, void * __p) {
  // X86_64-LABEL: @test_wrussq
  // X86_64:       call void @llvm.x86.wrussq(i64 %{{[a-z0-9.]+}}, i8* %{{[a-z0-9.]+}})
  _wrussq(__a, __p);
}
#endif

void test_setssbsy() {
  // CHECK-LABEL: @test_setssbsy
  // CHECK:       call void @llvm.x86.setssbsy()
  _setssbsy();
}

void test_clrssbsy(void * __p) {
  // CHECK-LABEL: @test_clrssbsy
  // CHECK:       call void @llvm.x86.clrssbsy(i8* %{{[a-z0-9.]+}})
  _clrssbsy(__p);
}
