// Test for debug info for C++11 deleted member functions

//Supported: -O0, standalone DI
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu %s -o - \
// RUN:   -O0 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone \
// RUN: | FileCheck %s -check-prefix=ATTR

// ATTR: DISubprogram(name: "deleted", {{.*}}, flags: DIFlagPublic | DIFlagPrototyped,
// ATTR: DISubprogram(name: "deleted", {{.*}}, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted
// ATTR: DISubprogram(name: "operator=", linkageName: "_ZN7deletedaSERKS_", {{.*}}, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted
// ATTR: DISubprogram(name: "deleted", {{.*}}, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted
// ATTR: DISubprogram(name: "operator=", linkageName: "_ZN7deletedaSEOS_", {{.*}}, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagDeleted
// ATTR: DISubprogram(name: "~deleted", {{.*}}, flags: DIFlagPublic | DIFlagPrototyped,
class deleted {
public:
  // Defaulted on purpose, so as to facilitate object creation
  deleted() = default;

  deleted(const deleted &) = delete;
  deleted &operator=(const deleted &) = delete;

  deleted(deleted &&) = delete;
  deleted &operator=(deleted &&) = delete;

  ~deleted() = default;
};

void foo() {
  deleted obj1;
}
