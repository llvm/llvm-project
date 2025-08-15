// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c++ -std=c++17 %s -debug-info-kind=line-tables-only -emit-llvm -o - -gno-column-info \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank
// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -gno-column-info \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check that atomic handling code gets Key Instruction annotations.

_Atomic(unsigned int) x;
unsigned int y;
void fun() {
  unsigned int r1 = __c11_atomic_fetch_add(&x,- 1, __ATOMIC_RELAXED);
// CHECK:      store i32 -1, ptr %.atomictmp, align 4,                !dbg [[LINE11_G2R1:!.*]]
// CHECK-NEXT: %0 = load i32, ptr %.atomictmp, align 4,               !dbg [[LINE11:!.*]]
// CHECK-NEXT: %1 = atomicrmw add ptr @x, i32 %0 monotonic, align 4,  !dbg [[LINE11_G2R2:!.*]]
// CHECK-NEXT: store i32 %1, ptr %atomic-temp, align 4,               !dbg [[LINE11_G2R1]]
// CHECK-NEXT: %2 = load i32, ptr %atomic-temp, align 4,              !dbg [[LINE11_G1R2:!.*]]
// CHECK-NEXT: store i32 %2, ptr %r1, align 4,                        !dbg [[LINE11_G1R1:!.*]]

  unsigned int r2 = __c11_atomic_load(&x, __ATOMIC_RELAXED);
// CHECK-NEXT: %3 = load atomic i32, ptr @x monotonic, align 4,       !dbg [[LINE19_G4R2:!.*]]
// CHECK-NEXT: store i32 %3, ptr %atomic-temp1, align 4,              !dbg [[LINE19_G4R1:!.*]]
// CHECK-NEXT: %4 = load i32, ptr %atomic-temp1, align 4,             !dbg [[LINE19_G3R2:!.*]]
// CHECK-NEXT: store i32 %4, ptr %r2, align 4,                        !dbg [[LINE19_G3R1:!.*]]

  __c11_atomic_store(&x, 2, __ATOMIC_RELAXED);
// CHECK-NEXT: store i32 2, ptr %.atomictmp2, align 4,                !dbg [[LINE25_G5R1:!.*]]
// CHECK-NEXT: %5 = load i32, ptr %.atomictmp2, align 4,              !dbg [[LINE25_G5R2:!.*]]
// CHECK-NEXT: store atomic i32 %5, ptr @x monotonic, align 4,        !dbg [[LINE25_G5R1:!.*]]

  int r3 = __atomic_test_and_set(&x, __ATOMIC_RELAXED);
// CHECK-NEXT: %6 = atomicrmw xchg ptr @x, i8 1 monotonic, align 4,   !dbg [[LINE30:!.*]]
// CHECK-NEXT: %tobool = icmp ne i8 %6, 0,                            !dbg [[LINE30_G7R2:!.*]]
// CHECK-NEXT: store i1 %tobool, ptr %atomic-temp3, align 1,          !dbg [[LINE30_G7R1:!.*]]
// CHECK-NEXT: %7 = load i8, ptr %atomic-temp3, align 1,              !dbg [[LINE30_G6R4:!.*]]
// CHECK-NEXT: %loadedv = trunc i8 %7 to i1,                          !dbg [[LINE30_G6R3:!.*]]
// CHECK-NEXT: %conv = zext i1 %loadedv to i32,                       !dbg [[LINE30_G6R2:!.*]]
// CHECK-NEXT: store i32 %conv, ptr %r3, align 4,                     !dbg [[LINE30_G6R1:!.*]]

  __atomic_clear(&x, __ATOMIC_RELAXED);
// CHECK-NEXT: store atomic i8 0, ptr @x monotonic, align 4,          !dbg [[LINE39_G8R1:!.*]]

  int r4 = __c11_atomic_exchange(&x, 2,__ATOMIC_RELAXED);
// CHECK-NEXT: store i32 2, ptr %.atomictmp4, align 4,                !dbg [[LINE42_G10R1:!.*]]
// CHECK-NEXT: %8 = load i32, ptr %.atomictmp4, align 4,              !dbg [[LINE42:!.*]]
// CHECK-NEXT: %9 = atomicrmw xchg ptr @x, i32 %8 monotonic, align 4, !dbg [[LINE42_G10R2:!.*]]
// CHECK-NEXT: store i32 %9, ptr %atomic-temp5, align 4,              !dbg [[LINE42_G10R1:!.*]]
// CHECK-NEXT: %10 = load i32, ptr %atomic-temp5, align 4,            !dbg [[LINE42_G9R2:!.*]]
// CHECK-NEXT: store i32 %10, ptr %r4, align 4,                       !dbg [[LINE42_G9R1:!.*]]

  int r5 = __atomic_compare_exchange(&y, &y, &y, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
// CHECK-NEXT: %11 = load i32, ptr @y, align 4,                       !dbg [[LINE50:!.*]]
// CHECK-NEXT: %12 = load i32, ptr @y, align 4,                       !dbg [[LINE50]]
// CHECK-NEXT: %13 = cmpxchg ptr @y, i32 %11, i32 %12 monotonic monotonic, align 4, !dbg [[LINE50]]
// CHECK-NEXT: %14 = extractvalue { i32, i1 } %13, 0,                 !dbg [[LINE50_G12R2:!.*]]
// CHECK-NEXT: %15 = extractvalue { i32, i1 } %13, 1,                 !dbg [[LINE50_G12R3:!.*]]
// CHECK-NEXT: br i1 %15, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg [[LINE50]]
// CHECK: cmpxchg.store_expected:
// CHECK-NEXT: store i32 %14, ptr @y, align 4,                        !dbg [[LINE50_G12R1:!.*]]
// CHECK-NEXT: br label %cmpxchg.continue,                            !dbg [[LINE50]]
// CHECK: cmpxchg.continue:
// CHECK-NEXT: %storedv = zext i1 %15 to i8,                          !dbg [[LINE50_G12R2]]
// CHECK-NEXT: store i8 %storedv, ptr %cmpxchg.bool, align 1,         !dbg [[LINE50_G12R1]]
// CHECK-NEXT: %16 = load i8, ptr %cmpxchg.bool, align 1,             !dbg [[LINE50_G11R4:!.*]]
// CHECK-NEXT: %loadedv6 = trunc i8 %16 to i1,                        !dbg [[LINE50_G11R3:!.*]]
// CHECK-NEXT: %conv7 = zext i1 %loadedv6 to i32,                     !dbg [[LINE50_G11R2:!.*]]
// CHECK-NEXT: store i32 %conv7, ptr %r5, align 4,                    !dbg [[LINE50_G11R1:!.*]]

  int r6 = __c11_atomic_compare_exchange_strong(&x, &y, 42, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
// CHECK-NEXT: store i32 42, ptr %.atomictmp8, align 4,               !dbg [[LINE68_G14R1:!.*]]
// CHECK-NEXT: %17 = load i32, ptr @y, align 4,                       !dbg [[LINE68:!.*]]
// CHECK-NEXT: %18 = load i32, ptr %.atomictmp8, align 4,             !dbg [[LINE68]]
// CHECK-NEXT: %19 = cmpxchg ptr @x, i32 %17, i32 %18 monotonic monotonic, align 4, !dbg [[LINE68]]
// CHECK-NEXT: %20 = extractvalue { i32, i1 } %19, 0,                 !dbg [[LINE68_G14R2:!.*]]
// CHECK-NEXT: %21 = extractvalue { i32, i1 } %19, 1,                 !dbg [[LINE68_G14R3:!.*]]
// CHECK-NEXT: br i1 %21, label %cmpxchg.continue11, label %cmpxchg.store_expected10, !dbg [[LINE68]]
// CHECK: cmpxchg.store_expected10:
// CHECK-NEXT: store i32 %20, ptr @y, align 4,                        !dbg [[LINE68_G14R1:!.*]]
// CHECK-NEXT: br label %cmpxchg.continue11,                          !dbg [[LINE68]]
// CHECK: cmpxchg.continue11:
// CHECK-NEXT: %storedv12 = zext i1 %21 to i8,                        !dbg [[LINE68_G14R2]]
// CHECK-NEXT: store i8 %storedv12, ptr %cmpxchg.bool9, align 1,      !dbg [[LINE68_G14R1:!.*]]
// CHECK-NEXT: %22 = load i8, ptr %cmpxchg.bool9, align 1,            !dbg [[LINE68_G13R4:!.*]]
// CHECK-NEXT: %loadedv13 = trunc i8 %22 to i1,                       !dbg [[LINE68_G13R3:!.*]]
// CHECK-NEXT: %conv14 = zext i1 %loadedv13 to i32,                   !dbg [[LINE68_G13R2:!.*]]
// CHECK-NEXT: store i32 %conv14, ptr %r6, align 4,                   !dbg [[LINE68_G13R1:!.*]]

  int r7 = __c11_atomic_compare_exchange_weak(&x, &y, 43, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
// CHECK-NEXT: store i32 43, ptr %.atomictmp15, align 4,              !dbg [[LINE87_G16R1:!.*]]
// CHECK-NEXT: %23 = load i32, ptr @y, align 4,                       !dbg [[LINE87:!.*]]
// CHECK-NEXT: %24 = load i32, ptr %.atomictmp15, align 4,            !dbg [[LINE87]]
// CHECK-NEXT: %25 = cmpxchg weak ptr @x, i32 %23, i32 %24 monotonic monotonic, align 4, !dbg [[LINE87]]
// CHECK-NEXT: %26 = extractvalue { i32, i1 } %25, 0,                 !dbg [[LINE87_G16R2:!.*]]
// CHECK-NEXT: %27 = extractvalue { i32, i1 } %25, 1,                 !dbg [[LINE87_G16R3:!.*]]
// CHECK-NEXT: br i1 %27, label %cmpxchg.continue18, label %cmpxchg.store_expected17, !dbg [[LINE87]]
// CHECK: cmpxchg.store_expected17:
// CHECK-NEXT: store i32 %26, ptr @y, align 4,                        !dbg [[LINE87_G16R1]]
// CHECK-NEXT: br label %cmpxchg.continue18,                          !dbg [[LINE87]]
// CHECK: cmpxchg.continue18:
// CHECK-NEXT: %storedv19 = zext i1 %27 to i8,                        !dbg [[LINE87_G16R2]]
// CHECK-NEXT: store i8 %storedv19, ptr %cmpxchg.bool16, align 1,     !dbg [[LINE87_G16R1]]
// CHECK-NEXT: %28 = load i8, ptr %cmpxchg.bool16, align 1,           !dbg [[LINE87_G15R4:!.*]]
// CHECK-NEXT: %loadedv20 = trunc i8 %28 to i1,                       !dbg [[LINE87_G15R3:!.*]]
// CHECK-NEXT: %conv21 = zext i1 %loadedv20 to i32,                   !dbg [[LINE87_G15R2:!.*]]
// CHECK-NEXT: store i32 %conv21, ptr %r7, align 4,                   !dbg [[LINE87_G15R1:!.*]]

// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[LINE11_G2R1]] = !DILocation(line: 11, scope: ![[#]], atomGroup: 2, atomRank: 1)
// CHECK: [[LINE11]]      = !DILocation(line: 11, scope: ![[#]])
// CHECK: [[LINE11_G2R2]] = !DILocation(line: 11, scope: ![[#]], atomGroup: 2, atomRank: 2)
// CHECK: [[LINE11_G1R2]] = !DILocation(line: 11, scope: ![[#]], atomGroup: 1, atomRank: 2)
// CHECK: [[LINE11_G1R1]] = !DILocation(line: 11, scope: ![[#]], atomGroup: 1, atomRank: 1)

// CHECK: [[LINE19_G4R2]] = !DILocation(line: 19, scope: ![[#]], atomGroup: 4, atomRank: 2)
// CHECK: [[LINE19_G4R1]] = !DILocation(line: 19, scope: ![[#]], atomGroup: 4, atomRank: 1)
// CHECK: [[LINE19_G3R2]] = !DILocation(line: 19, scope: ![[#]], atomGroup: 3, atomRank: 2)
// CHECK: [[LINE19_G3R1]] = !DILocation(line: 19, scope: ![[#]], atomGroup: 3, atomRank: 1)

// CHECK: [[LINE25_G5R1]] = !DILocation(line: 25, scope: ![[#]], atomGroup: 5, atomRank: 1)
// CHECK: [[LINE25_G5R2]] = !DILocation(line: 25, scope: ![[#]], atomGroup: 5, atomRank: 2)

// CHECK: [[LINE30]]      = !DILocation(line: 30, scope: ![[#]])
// CHECK: [[LINE30_G7R2]] = !DILocation(line: 30, scope: ![[#]], atomGroup: 7, atomRank: 2)
// CHECK: [[LINE30_G7R1]] = !DILocation(line: 30, scope: ![[#]], atomGroup: 7, atomRank: 1)
// CHECK: [[LINE30_G6R4]] = !DILocation(line: 30, scope: ![[#]], atomGroup: 6, atomRank: 4)
// CHECK: [[LINE30_G6R3]] = !DILocation(line: 30, scope: ![[#]], atomGroup: 6, atomRank: 3)
// CHECK: [[LINE30_G6R2]] = !DILocation(line: 30, scope: ![[#]], atomGroup: 6, atomRank: 2)
// CHECK: [[LINE30_G6R1]] = !DILocation(line: 30, scope: ![[#]], atomGroup: 6, atomRank: 1)

// CHECK: [[LINE39_G8R1]] = !DILocation(line: 39, scope: ![[#]], atomGroup: 8, atomRank: 1)

// CHECK: [[LINE42_G10R1]] = !DILocation(line: 42, scope: ![[#]], atomGroup: 10, atomRank: 1)
// CHECK: [[LINE42]]       = !DILocation(line: 42, scope: ![[#]])
// CHECK: [[LINE42_G10R2]] = !DILocation(line: 42, scope: ![[#]], atomGroup: 10, atomRank: 2)
// CHECK: [[LINE42_G9R2]]  = !DILocation(line: 42, scope: ![[#]], atomGroup: 9, atomRank: 2)
// CHECK: [[LINE42_G9R1]]  = !DILocation(line: 42, scope: ![[#]], atomGroup: 9, atomRank: 1)

// CHECK: [[LINE50]]       = !DILocation(line: 50, scope: ![[#]])
// CHECK: [[LINE50_G12R2]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 12, atomRank: 2)
// CHECK: [[LINE50_G12R3]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 12, atomRank: 3)
// CHECK: [[LINE50_G12R1]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 12, atomRank: 1)
// CHECK: [[LINE50_G11R4]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 11, atomRank: 4)
// CHECK: [[LINE50_G11R3]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 11, atomRank: 3)
// CHECK: [[LINE50_G11R2]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 11, atomRank: 2)
// CHECK: [[LINE50_G11R1]] = !DILocation(line: 50, scope: ![[#]], atomGroup: 11, atomRank: 1)

// CHECK: [[LINE68_G14R1]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 14, atomRank: 1)
// CHECK: [[LINE68]]       = !DILocation(line: 68, scope: ![[#]])
// CHECK: [[LINE68_G14R2]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 14, atomRank: 2)
// CHECK: [[LINE68_G14R3]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 14, atomRank: 3)
// CHECK: [[LINE68_G13R4]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 13, atomRank: 4)
// CHECK: [[LINE68_G13R3]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 13, atomRank: 3)
// CHECK: [[LINE68_G13R2]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 13, atomRank: 2)
// CHECK: [[LINE68_G13R1]] = !DILocation(line: 68, scope: ![[#]], atomGroup: 13, atomRank: 1)

// CHECK: [[LINE87_G16R1]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 16, atomRank: 1)
// CHECK: [[LINE87]]       = !DILocation(line: 87, scope: ![[#]])
// CHECK: [[LINE87_G16R2]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 16, atomRank: 2)
// CHECK: [[LINE87_G16R3]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 16, atomRank: 3)
// CHECK: [[LINE87_G15R4]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 15, atomRank: 4)
// CHECK: [[LINE87_G15R3]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 15, atomRank: 3)
// CHECK: [[LINE87_G15R2]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 15, atomRank: 2)
// CHECK: [[LINE87_G15R1]] = !DILocation(line: 87, scope: ![[#]], atomGroup: 15, atomRank: 1)

// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 17, atomRank: 1)
