// Tests that we emit unified constructor/destructor linkage names
// for ABIs that support it.

// Check that -gstructor-decl-linkage-names is the default.
// RUN: %clang_cc1 -triple aarch64-apple-macosx -emit-llvm -debug-info-kind=standalone \
// RUN:            %s -o - | FileCheck %s --check-prefixes=CHECK,ITANIUM
//
// Check with -gstructor-decl-linkage-names.
// RUN: %clang_cc1 -triple aarch64-apple-macosx -emit-llvm -debug-info-kind=standalone \
// RUN:            -gstructor-decl-linkage-names %s -o - | FileCheck %s --check-prefixes=CHECK,ITANIUM
//
// Check with -gno-structor-decl-linkage-names.
// RUN: %clang_cc1 -triple aarch64-apple-macosx -emit-llvm -debug-info-kind=standalone \
// RUN:            -gno-structor-decl-linkage-names %s -o - | FileCheck %s --check-prefixes=CHECK,DISABLE
//
// Check ABI without structor variants.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -debug-info-kind=standalone \
// RUN:            -gstructor-decl-linkage-names %s -o - | FileCheck %s --check-prefixes=CHECK,MSABI

struct Base {
  Base(int x);
  ~Base();
};

Base::Base(int x) {}
Base::~Base() {}

// Check that we emit unified ctor/dtor (C4/D4) on Itanium but not for MS-ABI.

// CHECK: ![[BASE_CTOR_DECL:[0-9]+]] = !DISubprogram(name: "Base"
// MSABI-NOT:                                        linkageName:
// DISABLE-NOT:                                      linkageName:
// ITANIUM-SAME:                                     linkageName: "_ZN4BaseC4Ei"
// CHECK-SAME:                                       spFlags: 0

// CHECK: ![[BASE_DTOR_DECL:[0-9]+]] = !DISubprogram(name: "~Base"
// MSABI-NOT:                                        linkageName:
// DISABLE-NOT:                                      linkageName:
// ITANIUM-SAME:                                     linkageName: "_ZN4BaseD4Ev"
// CHECK-SAME:                                       spFlags: 0

// Check that the ctor/dtor definitions have linkage names that aren't
// the ones on the declaration.

// CHECK: !DISubprogram(name: "Base"
// MSABI-SAME:          linkageName:
// ITANIUM-SAME:        linkageName: "_ZN4BaseC2Ei"
// CHECK-SAME:          spFlags: DISPFlagDefinition
// CHECK-SAME:          declaration: ![[BASE_CTOR_DECL]]

// ITANIUM: !DISubprogram(name: "Base"
// ITANIUM-SAME:          linkageName: "_ZN4BaseC1Ei"
// ITANIUM-SAME:          spFlags: DISPFlagDefinition
// ITANIUM-SAME:          declaration: ![[BASE_CTOR_DECL]]

// CHECK: !DISubprogram(name: "~Base"
// MSABI-SAME:          linkageName:
// ITANIUM-SAME:        linkageName: "_ZN4BaseD2Ev"
// CHECK-SAME:          spFlags: DISPFlagDefinition
// CHECK-SAME:          declaration: ![[BASE_DTOR_DECL]]

// ITANIUM: !DISubprogram(name: "~Base"
// ITANIUM-SAME:          linkageName: "_ZN4BaseD1Ev"
// ITANIUM-SAME:          spFlags: DISPFlagDefinition
// ITANIUM-SAME:          declaration: ![[BASE_DTOR_DECL]]

struct Derived : public Base {
    using Base::Base;
} d(5);

// CHECK: !DISubprogram(name: "Base"
// MSABI-SAME:          linkageName:
// ITANIUM-SAME:        linkageName: "_ZN7DerivedCI14BaseEi"
// CHECK-SAME:          spFlags: {{.*}}DISPFlagDefinition
// CHECK-SAME:          declaration: ![[BASE_INHERIT_CTOR_DECL:[0-9]+]]

// CHECK: [[BASE_INHERIT_CTOR_DECL]] = !DISubprogram(name: "Base"
// MSABI-NOT:                                        linkageName:
// DISABLE-NOT:                                      linkageName:
// ITANIUM-SAME:                                     linkageName: "_ZN7DerivedCI44BaseEi"
// CHECK-SAME                                        spFlags: 0

// ITANIUM: !DISubprogram(name: "Base"
// ITANIUM-SAME:          linkageName: "_ZN7DerivedCI24BaseEi"
// ITANIUM-SAME:          spFlags: DISPFlagDefinition
// ITANIUM-SAME:          declaration: ![[BASE_INHERIT_CTOR_DECL:[0-9]+]]

// MSABI:   !DISubprogram(name: "~Derived"
// DISABLE: !DISubprogram(name: "~Derived"
