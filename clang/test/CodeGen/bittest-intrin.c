// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc %s -emit-llvm -o - | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -fms-extensions -triple thumbv7-windows-msvc %s -emit-llvm -o - | FileCheck %s --check-prefix=ARM
// RUN: %clang_cc1 -fms-extensions -triple aarch64-windows-msvc %s -emit-llvm -o - | FileCheck %s --check-prefix=ARM

volatile unsigned char sink = 0;
void test32(long *base, long idx) {
  sink = _bittest(base, idx);
  sink = _bittestandcomplement(base, idx);
  sink = _bittestandreset(base, idx);
  sink = _bittestandset(base, idx);
  sink = _interlockedbittestandreset(base, idx);
  sink = _interlockedbittestandset(base, idx);
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

#if defined(_M_ARM) || defined(_M_ARM64)
void test_arm(long *base, long idx) {
  sink = _interlockedbittestandreset_acq(base, idx);
  sink = _interlockedbittestandreset_rel(base, idx);
  sink = _interlockedbittestandreset_nf(base, idx);
  sink = _interlockedbittestandset_acq(base, idx);
  sink = _interlockedbittestandset_rel(base, idx);
  sink = _interlockedbittestandset_nf(base, idx);
}
#endif

// X64-LABEL: define dso_local void @test32(ptr noundef %base, i32 noundef %idx)
// X64: call i8 asm sideeffect "btl $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i32 {{.*}})
// X64: call i8 asm sideeffect "btcl $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i32 {{.*}})
// X64: call i8 asm sideeffect "btrl $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i32 {{.*}})
// X64: call i8 asm sideeffect "btsl $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i32 {{.*}})
// X64: call i8 asm sideeffect "lock btrl $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i32 {{.*}})
// X64: call i8 asm sideeffect "lock btsl $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i32 {{.*}})

// X64-LABEL: define dso_local void @test64(ptr noundef %base, i64 noundef %idx)
// X64: call i8 asm sideeffect "btq $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i64 {{.*}})
// X64: call i8 asm sideeffect "btcq $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i64 {{.*}})
// X64: call i8 asm sideeffect "btrq $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i64 {{.*}})
// X64: call i8 asm sideeffect "btsq $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i64 {{.*}})
// X64: call i8 asm sideeffect "lock btrq $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i64 {{.*}})
// X64: call i8 asm sideeffect "lock btsq $2, ($1)", "={@ccc},r,r,~{cc},~{memory},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}}, i64 {{.*}})

// ARM-LABEL: define dso_local {{.*}}void @test32(ptr noundef %base, i32 noundef %idx)
// ARM: %[[IDXHI:[^ ]*]] = ashr i32 %{{.*}}, 3
// ARM: %[[BYTEADDR:[^ ]*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[IDXHI]]
// ARM: %[[IDX8:[^ ]*]] = trunc i32 %{{.*}} to i8
// ARM: %[[IDXLO:[^ ]*]] = and i8 %[[IDX8]], 7
// ARM: %[[BYTE:[^ ]*]] = load i8, ptr %[[BYTEADDR]], align 1
// ARM: %[[BYTESHR:[^ ]*]] = lshr i8 %[[BYTE]], %[[IDXLO]]
// ARM: %[[RES:[^ ]*]] = and i8 %[[BYTESHR]], 1
// ARM: store volatile i8 %[[RES]], ptr @sink, align 1

// ARM: %[[IDXHI:[^ ]*]] = ashr i32 %{{.*}}, 3
// ARM: %[[BYTEADDR:[^ ]*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[IDXHI]]
// ARM: %[[IDX8:[^ ]*]] = trunc i32 %{{.*}} to i8
// ARM: %[[IDXLO:[^ ]*]] = and i8 %[[IDX8]], 7
// ARM: %[[MASK:[^ ]*]] = shl i8 1, %[[IDXLO]]
// ARM: %[[BYTE:[^ ]*]] = load i8, ptr %[[BYTEADDR]], align 1
// ARM: %[[NEWBYTE:[^ ]*]] = xor i8 %[[BYTE]], %[[MASK]]
// ARM: store i8 %[[NEWBYTE]], ptr %[[BYTEADDR]], align 1
// ARM: %[[BYTESHR:[^ ]*]] = lshr i8 %[[BYTE]], %[[IDXLO]]
// ARM: %[[RES:[^ ]*]] = and i8 %[[BYTESHR]], 1
// ARM: store volatile i8 %[[RES]], ptr @sink, align 1

// ARM: %[[IDXHI:[^ ]*]] = ashr i32 %{{.*}}, 3
// ARM: %[[BYTEADDR:[^ ]*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[IDXHI]]
// ARM: %[[IDX8:[^ ]*]] = trunc i32 %{{.*}} to i8
// ARM: %[[IDXLO:[^ ]*]] = and i8 %[[IDX8]], 7
// ARM: %[[MASK:[^ ]*]] = shl i8 1, %[[IDXLO]]
// ARM: %[[BYTE:[^ ]*]] = load i8, ptr %[[BYTEADDR]], align 1
// ARM: %[[NOTMASK:[^ ]*]] = xor i8 %[[MASK]], -1
// ARM: %[[NEWBYTE:[^ ]*]] = and i8 %[[BYTE]], %[[NOTMASK]]
// ARM: store i8 %[[NEWBYTE]], ptr %[[BYTEADDR]], align 1
// ARM: %[[BYTESHR:[^ ]*]] = lshr i8 %[[BYTE]], %[[IDXLO]]
// ARM: %[[RES:[^ ]*]] = and i8 %[[BYTESHR]], 1
// ARM: store volatile i8 %[[RES]], ptr @sink, align 1

// ARM: %[[IDXHI:[^ ]*]] = ashr i32 %{{.*}}, 3
// ARM: %[[BYTEADDR:[^ ]*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[IDXHI]]
// ARM: %[[IDX8:[^ ]*]] = trunc i32 %{{.*}} to i8
// ARM: %[[IDXLO:[^ ]*]] = and i8 %[[IDX8]], 7
// ARM: %[[MASK:[^ ]*]] = shl i8 1, %[[IDXLO]]
// ARM: %[[BYTE:[^ ]*]] = load i8, ptr %[[BYTEADDR]], align 1
// ARM: %[[NEWBYTE:[^ ]*]] = or i8 %[[BYTE]], %[[MASK]]
// ARM: store i8 %[[NEWBYTE]], ptr %[[BYTEADDR]], align 1
// ARM: %[[BYTESHR:[^ ]*]] = lshr i8 %[[BYTE]], %[[IDXLO]]
// ARM: %[[RES:[^ ]*]] = and i8 %[[BYTESHR]], 1
// ARM: store volatile i8 %[[RES]], ptr @sink, align 1

// ARM: %[[IDXHI:[^ ]*]] = ashr i32 %{{.*}}, 3
// ARM: %[[BYTEADDR:[^ ]*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[IDXHI]]
// ARM: %[[IDX8:[^ ]*]] = trunc i32 %{{.*}} to i8
// ARM: %[[IDXLO:[^ ]*]] = and i8 %[[IDX8]], 7
// ARM: %[[MASK:[^ ]*]] = shl i8 1, %[[IDXLO]]
// ARM: %[[NOTMASK:[^ ]*]] = xor i8 %[[MASK]], -1
// ARM: %[[BYTE:[^ ]*]] = atomicrmw and ptr %[[BYTEADDR]], i8 %[[NOTMASK]] seq_cst, align 1
// ARM: %[[BYTESHR:[^ ]*]] = lshr i8 %[[BYTE]], %[[IDXLO]]
// ARM: %[[RES:[^ ]*]] = and i8 %[[BYTESHR]], 1
// ARM: store volatile i8 %[[RES]], ptr @sink, align 1

// ARM: %[[IDXHI:[^ ]*]] = ashr i32 %{{.*}}, 3
// ARM: %[[BYTEADDR:[^ ]*]] = getelementptr inbounds i8, ptr %{{.*}}, i32 %[[IDXHI]]
// ARM: %[[IDX8:[^ ]*]] = trunc i32 %{{.*}} to i8
// ARM: %[[IDXLO:[^ ]*]] = and i8 %[[IDX8]], 7
// ARM: %[[MASK:[^ ]*]] = shl i8 1, %[[IDXLO]]
// ARM: %[[BYTE:[^ ]*]] = atomicrmw or ptr %[[BYTEADDR]], i8 %[[MASK]] seq_cst, align 1
// ARM: %[[BYTESHR:[^ ]*]] = lshr i8 %[[BYTE]], %[[IDXLO]]
// ARM: %[[RES:[^ ]*]] = and i8 %[[BYTESHR]], 1
// ARM: store volatile i8 %[[RES]], ptr @sink, align 1


// Just look for the atomicrmw instructions.

// ARM-LABEL: define dso_local {{.*}}void @test_arm(ptr noundef %base, i32 noundef %idx)
// ARM: atomicrmw and ptr %{{.*}}, i8 {{.*}} acquire, align 1
// ARM: atomicrmw and ptr %{{.*}}, i8 {{.*}} release, align 1
// ARM: atomicrmw and ptr %{{.*}}, i8 {{.*}} monotonic, align 1
// ARM: atomicrmw or ptr %{{.*}}, i8 {{.*}} acquire, align 1
// ARM: atomicrmw or ptr %{{.*}}, i8 {{.*}} release, align 1
// ARM: atomicrmw or ptr %{{.*}}, i8 {{.*}} monotonic, align 1
