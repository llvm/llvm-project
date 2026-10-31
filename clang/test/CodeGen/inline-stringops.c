// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - -finline-stringops=memset %s | FileCheck --check-prefix=CHECK-DEF-ATTR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - -finline-stringops %s | FileCheck --check-prefix=CHECK-DEF-ATTR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-NO-ATTR %s

extern void bar(void);

void foo(void) {
  bar();
}

// CHECK-DEF-ATTR: define dso_local void @foo() #[[ATTR:[0-9]+]]
// CHECK-DEF-ATTR: declare void @bar() #[[DECL_ATTR:[0-9]+]]
// CHECK-DEF-ATTR: attributes #[[ATTR]] = { {{.*}}"inline-memset"="true"{{.*}} }
// CHECK-DEF-ATTR-NOT: attributes #[[DECL_ATTR]] = { {{.*}}"inline-memset"

// CHECK-NO-ATTR-NOT: "inline-memset"
