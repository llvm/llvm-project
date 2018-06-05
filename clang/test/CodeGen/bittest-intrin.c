// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc %s -emit-llvm -o - | FileCheck %s

volatile unsigned char sink = 0;
void test32(long *base, long idx) {
  sink = _bittest(base, idx);
  sink = _bittestandcomplement(base, idx);
  sink = _bittestandreset(base, idx);
  sink = _bittestandset(base, idx);
  sink = _interlockedbittestandreset(base, idx);
  sink = _interlockedbittestandset(base, idx);
}
void test64(__int64 *base, __int64 idx) {
  sink = _bittest64(base, idx);
  sink = _bittestandcomplement64(base, idx);
  sink = _bittestandreset64(base, idx);
  sink = _bittestandset64(base, idx);
  sink = _interlockedbittestandreset64(base, idx);
  sink = _interlockedbittestandset64(base, idx);
}

// CHECK-LABEL: define dso_local void @test32(i32* %base, i32 %idx)
// CHECK: call i8 asm sideeffect "btl $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i32* %{{.*}}, i32 {{.*}})
// CHECK: call i8 asm sideeffect "btcl $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i32* %{{.*}}, i32 {{.*}})
// CHECK: call i8 asm sideeffect "btrl $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i32* %{{.*}}, i32 {{.*}})
// CHECK: call i8 asm sideeffect "btsl $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i32* %{{.*}}, i32 {{.*}})
// CHECK: call i8 asm sideeffect "lock btrl $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i32* %{{.*}}, i32 {{.*}})
// CHECK: call i8 asm sideeffect "lock btsl $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i32* %{{.*}}, i32 {{.*}})

// CHECK-LABEL: define dso_local void @test64(i64* %base, i64 %idx)
// CHECK: call i8 asm sideeffect "btq $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i64* %{{.*}}, i64 {{.*}})
// CHECK: call i8 asm sideeffect "btcq $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i64* %{{.*}}, i64 {{.*}})
// CHECK: call i8 asm sideeffect "btrq $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i64* %{{.*}}, i64 {{.*}})
// CHECK: call i8 asm sideeffect "btsq $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i64* %{{.*}}, i64 {{.*}})
// CHECK: call i8 asm sideeffect "lock btrq $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i64* %{{.*}}, i64 {{.*}})
// CHECK: call i8 asm sideeffect "lock btsq $2, ($1)\0A\09setc ${0:b}", "=r,r,r,~{{.*}}"(i64* %{{.*}}, i64 {{.*}})
