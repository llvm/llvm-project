// RUN: %clang_cc1 -ffreestanding -fms-extensions -triple x86_64-unknown-linux-gnu \
// RUN:         -Oz -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

// RUN: %clang_cc1 -ffreestanding -fms-extensions -triple x86_64-unknown-linux-gnu \
// RUN:         -Oz -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// RUN: %clang_cc1 -ffreestanding -fms-extensions -triple x86_64-unknown-linux-gnu \
// RUN:         -Oz -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

unsigned __int64 __shiftleft128(unsigned __int64 low, unsigned __int64 high,
                                unsigned char shift);
unsigned __int64 __shiftright128(unsigned __int64 low, unsigned __int64 high,
                                 unsigned char shift);

// CIR-LABEL: cir.func{{.*}}@test_shiftleft128
// CIR: (%[[L:[^,]+]]: !u64i{{.*}}, %[[H:[^,]+]]: !u64i{{.*}}, %[[D:[^,]+]]: !u8i{{.*}})
// CIR: %[[D_LOAD:[^ ]+]] = cir.load {{.*}} %{{[^ ]+}} : !cir.ptr<!u8i>, !u8i
// CIR: %[[D_CAST:[^ ]+]] = cir.cast integral %[[D_LOAD]] : !u8i -> !u64i
// CIR: %{{[^ ]+}} = cir.call_llvm_intrinsic "fshl" {{.*}} : (!u64i, !u64i, !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: define dso_local i64 @test_shiftleft128(i64 %0, i64 %1, i8 %2)
// LLVM-NEXT: [[TMP1:%.*]] = zext i8 %2 to i64
// LLVM-NEXT: [[TMP2:%.*]] = tail call i64 @llvm.fshl.i64(i64 %1, i64 %0, i64 [[TMP1]])
// LLVM-NEXT: ret i64 [[TMP2]]

// OGCG-LABEL: define dso_local noundef i64 @test_shiftleft128(i64 noundef %l, i64 noundef %h, i8 noundef zeroext %d)
// OGCG-NEXT: entry:
// OGCG-NEXT: [[TMP0:%.*]] = zext i8 %d to i64
// OGCG-NEXT: [[TMP1:%.*]] = tail call i64 @llvm.fshl.i64(i64 %h, i64 %l, i64 [[TMP0]])
// OGCG-NEXT: ret i64 [[TMP1]]
unsigned __int64 test_shiftleft128(unsigned __int64 l, unsigned __int64 h,
                                   unsigned char d) {
  return __shiftleft128(l, h, d);
}

// CIR-LABEL: cir.func{{.*}}@test_shiftright128
// CIR: (%[[L:[^,]+]]: !u64i{{.*}}, %[[H:[^,]+]]: !u64i{{.*}}, %[[D:[^,]+]]: !u8i{{.*}})
// CIR: %[[D_LOAD:[^ ]+]] = cir.load {{.*}} %{{[^ ]+}} : !cir.ptr<!u8i>, !u8i
// CIR: %[[D_CAST:[^ ]+]] = cir.cast integral %[[D_LOAD]] : !u8i -> !u64i
// CIR: %{{[^ ]+}} = cir.call_llvm_intrinsic "fshr" {{.*}} : (!u64i, !u64i, !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: define dso_local i64 @test_shiftright128(i64 %0, i64 %1, i8 %2)
// LLVM-NEXT: [[TMP1:%.*]] = zext i8 %2 to i64
// LLVM-NEXT: [[TMP2:%.*]] = tail call i64 @llvm.fshr.i64(i64 %1, i64 %0, i64 [[TMP1]])
// LLVM-NEXT: ret i64 [[TMP2]]

// OGCG-LABEL: define dso_local noundef i64 @test_shiftright128(i64 noundef %l, i64 noundef %h, i8 noundef zeroext %d)
// OGCG-NEXT: entry:
// OGCG-NEXT: [[TMP0:%.*]] = zext i8 %d to i64
// OGCG-NEXT: [[TMP1:%.*]] = tail call i64 @llvm.fshr.i64(i64 %h, i64 %l, i64 [[TMP0]])
// OGCG-NEXT: ret i64 [[TMP1]]
unsigned __int64 test_shiftright128(unsigned __int64 l, unsigned __int64 h,
                                    unsigned char d) {
  return __shiftright128(l, h, d);
}
