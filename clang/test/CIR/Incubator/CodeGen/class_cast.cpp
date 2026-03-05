// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=OGCG
class Base {
  // CIR-LABEL: _ZN4BaseaSERS_
  // CIR-SAME: ([[ARG0:%.*]]: !cir.ptr{{.*}}, [[ARG1:%.*]]: !cir.ptr{{.*}})
  // CIR: [[ALLOCA_0:%.*]] =  cir.alloca
  // CIR: [[ALLOCA_1:%.*]] =  cir.alloca
  // CIR: [[ALLOCA_2:%.*]] =  cir.alloca
  // CIR: cir.store [[ARG0]], [[ALLOCA_0]]
  // CIR: cir.store [[ARG1]], [[ALLOCA_1]]
  // CIR: [[LD_0:%.*]] = cir.load deref [[ALLOCA_0]]
  // CIR: cir.store align(8) [[LD_0]], [[ALLOCA_2]]
  // CIR: [[LD_1:%.*]] = cir.load{{.*}} [[ALLOCA_2]]
  // CIR: cir.return [[LD_1]]

  // LLVM-LABEL: _ZN4BaseaSERS_
  // LLVM-SAME: (ptr [[ARG0:%.*]], ptr [[ARG1:%.*]]) 
  // LLVM:       [[TMP3:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT:  [[TMP4:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT:  [[TMP5:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT:  store ptr [[ARG0]], ptr [[TMP3]], align 8
  // LLVM-NEXT:  store ptr [[ARG1]], ptr [[TMP4]], align 8
  // LLVM-NEXT:  [[TMP6:%.*]] = load ptr, ptr [[TMP3]], align 8
  // LLVM-NEXT:  store ptr [[TMP6]], ptr [[TMP5]], align 8
  // LLVM-NEXT:  [[TMP7:%.*]] = load ptr, ptr [[TMP5]], align 8
  // LLVM-NEXT:  ret ptr [[TMP7]]

public:
  Base &operator=(Base &b) {
    return *this;
  }
};

class Derived : Base {
  Derived &operator=(Derived &);
};
Derived &Derived::operator=(Derived &B) {
  // CIR-LABEL: _ZN7DerivedaSERS_
  // CIR-SAME: [[ARG0:%.*]]: !cir.ptr{{.*}}, [[ARG1:%.*]]: !cir.ptr{{.*}}
  // CIR: cir.store [[ARG0]], [[ALLOCA_0:%.*]] :
  // CIR: cir.store [[ARG1]], [[ALLOCA_1:%.*]] :
  // CIR: [[LD_0:%.*]] = cir.load [[ALLOCA_0]]
  // CIR: [[BASE_ADDR_0:%.*]] = cir.base_class_addr [[LD_0]]
  // CIR: [[LD_1:%.*]] = cir.load [[ALLOCA_1]]
  // CIR: [[BASE_ADDR_1:%.*]] = cir.base_class_addr [[LD_1]]
  // CIR: [[CALL:%.*]] = cir.call @_ZN4BaseaSERS_
  // CIR: [[DERIVED_ADDR:%.*]] = cir.derived_class_addr [[CALL]]
  // CIR: cir.store{{.*}} [[DERIVED_ADDR]], [[ALLOCA_2:%.*]] :
  // CIR: [[LD_2:%.*]] = cir.load{{.*}} [[ALLOCA_2]]
  // CIR: cir.return [[LD_2]]

  // LLVM-LABEL: _ZN7DerivedaSERS_
  // LLVM-SAME: (ptr [[ARG0:%.*]], ptr [[ARG1:%.*]])
  // LLVM:       [[TMP3:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT:  [[TMP4:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT:  [[TMP5:%.*]] = alloca ptr, i64 1, align 8
  // LLVM-NEXT:  store ptr [[ARG0]], ptr [[TMP3]], align 8
  // LLVM-NEXT:  store ptr [[ARG1]], ptr [[TMP4]], align 8
  // LLVM-NEXT:  [[TMP6:%.*]] = load ptr, ptr [[TMP3]], align 8
  // LLVM-NEXT:  [[TMP7:%.*]] = load ptr, ptr [[TMP4]], align 8
  // LLVM-NEXT:  [[TMP8:%.*]] = call ptr @_ZN4BaseaSERS_(ptr [[TMP6]], ptr [[TMP7]])
  // LLVM-NEXT:  [[TMP9:%.*]] = getelementptr i8, ptr [[TMP8]], i32 0
  // LLVM-NEXT:  store ptr [[TMP9]], ptr [[TMP5]], align 8
  // LLVM-NEXT:  [[TMP10:%.*]] = load ptr, ptr [[TMP5]], align 8
  // LLVM-NEXT:  ret ptr [[TMP10]]

  // OGCG-LABEL: @_ZN7DerivedaSERS_
  // OGCG-SAME: (ptr{{.*}}[[ARG0:%.*]], ptr{{.*}}[[ARG1:%.*]])
  // OGCG:       [[TMP3:%.*]] = alloca ptr, align 8
  // OGCG-NEXT:  [[TMP4:%.*]] = alloca ptr, align 8
  // OGCG-NEXT:  store ptr [[ARG0]], ptr [[TMP3]], align 8
  // OGCG-NEXT:  store ptr [[ARG1]], ptr [[TMP4]], align 8
  // OGCG-NEXT:  [[TMP5:%.*]] = load ptr, ptr [[TMP3]], align 8
  // OGCG-NEXT:  [[TMP6:%.*]] = load ptr, ptr [[TMP4]], align 8
  // OGCG-NEXT:  [[TMP7:%.*]] = call{{.*}}ptr @_ZN4BaseaSERS_(ptr{{.*}}[[TMP5]], ptr{{.*}}[[TMP6]])
  // OGCG-NEXT:  ret ptr [[TMP7]]
  return (Derived &)Base::operator=(B);
}

// OGCG-LABEL: define{{.*}}@_ZN4BaseaSERS_
// OGCG-SAME: (ptr{{.*}}[[BASE_ARG0:%.*]], ptr{{.*}}[[BASE_ARG1:%.*]])
// OGCG:       [[TMP3:%.*]] = alloca ptr, align 8
// OGCG-NEXT:  [[TMP4:%.*]] = alloca ptr, align 8
// OGCG-NEXT:  store ptr [[BASE_ARG0]], ptr [[TMP3]], align 8 
// OGCG-NEXT:  store ptr [[BASE_ARG1]], ptr [[TMP4]], align 8
// OGCG-NEXT:  [[TMP5:%.*]] = load ptr, ptr [[TMP3]], align 8
// OGCG-NEXT:  ret ptr [[TMP5]]

