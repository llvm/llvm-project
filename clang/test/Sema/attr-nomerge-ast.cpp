// RUN: %clang_cc1 -ast-dump %s 2>&1 | FileCheck %s

[[clang::nomerge]] void func();
[[clang::nomerge]] void func();
void func();
[[clang::nomerge]] void func() {}
[[clang::nomerge]] void (*var)(void);

// CHECK: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: NoMergeAttr
// CHECK-NEXT: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: NoMergeAttr
// CHECK-NEXT: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: NoMergeAttr {{.*}} Inherited
// CHECK-NEXT: FunctionDecl {{.*}} func 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: NoMergeAttr

// CHECK-NEXT: VarDecl {{.*}} var 'void (*)()'
// CHECK-NEXT: NoMergeAttr
