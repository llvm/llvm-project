// Tests that we emit don't emit unified constructor/destructor linkage names
// for function-local constructors.

// Check with -gstructor-decl-linkage-names.
// RUN: %clang_cc1 -triple aarch64-apple-macosx -emit-llvm -debug-info-kind=standalone \
// RUN:            -gstructor-decl-linkage-names %s -o - | FileCheck %s --check-prefixes=CHECK
//
// Check with -gno-structor-decl-linkage-names.
// RUN: %clang_cc1 -triple aarch64-apple-macosx -emit-llvm -debug-info-kind=standalone \
// RUN:            -gno-structor-decl-linkage-names %s -o - | FileCheck %s --check-prefixes=CHECK

struct HasNestedCtor {
  HasNestedCtor();
};

HasNestedCtor::HasNestedCtor() {
  struct Local {
    Local() {}
    ~Local() {}
  } l;
}

// CHECK:      !DISubprogram(name: "Local"
// CHECK-NOT:                linkageName
// CHECK-SAME: )

// CHECK:      !DISubprogram(name: "~Local"
// CHECK-NOT:                linkageName
// CHECK-SAME: )
