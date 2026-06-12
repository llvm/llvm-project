// RUN: %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -fclangir -DBITCODE -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -fclangir -DBITCODE2 -emit-llvm-bc -o %t-2.bc %s
// RUN: %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -fclangir -mlink-bitcode-file %t.bc \
// RUN:     -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-BC %s
// RUN: %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -fclangir -mlink-builtin-bitcode %t.bc \
// RUN:     -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-BUILTIN-BC %s
// RUN: %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o - \
// RUN:     -mlink-bitcode-file %t.bc -mlink-bitcode-file %t-2.bc %s \
// RUN:     | FileCheck -check-prefix=CHECK-BC -check-prefix=CHECK-BC2 %s
// RUN: not %clang_cc1 -O1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:     -mlink-bitcode-file no-such-file.bc -emit-llvm -o - %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-NO-FILE %s
//
// -mlink-builtin-bitcode propagates the host TU's default function-definition
// attributes onto linked-in functions. Linking at -O0 keeps f from being
// inlined into g so the propagated attribute group is observable.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-cpu skylake -fclangir \
// RUN:     -mlink-builtin-bitcode %t.bc -emit-llvm -o - %s \
// RUN:     | FileCheck -check-prefix=CHECK-PROPAGATE %s
//
// -mlink-bitcode-file does NOT propagate (PropagateAttrs=false).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-cpu skylake -fclangir \
// RUN:     -mlink-bitcode-file %t.bc -emit-llvm -o - %s \
// RUN:     | FileCheck -check-prefix=CHECK-NO-PROPAGATE %s

int f(void);

#ifdef BITCODE

extern int f2(void);

volatile int gvar = 7;

int f(void) {
  f2();
  return 42;
}

#elif defined(BITCODE2)
int f2(void) { return 43; }
#else

extern volatile int gvar;

// -mlink-bitcode-file leaves linked symbols with external linkage.
// CHECK-BC: @gvar = global i32 7
// -mlink-builtin-bitcode internalizes linked symbols not referenced as roots.
// CHECK-BUILTIN-BC: @gvar = internal global i32 7

// CHECK-BC-LABEL: define{{.*}} i32 @g
// CHECK-BC: ret i32 42
// CHECK-BUILTIN-BC-LABEL: define{{.*}} i32 @g
// CHECK-BUILTIN-BC: ret i32 42
int g(void) {
  gvar = 1;
  return f();
}

// CHECK-BC-LABEL: define{{.*}} i32 @f
// CHECK-BC2-LABEL: define{{.*}} i32 @f2

// CHECK-PROPAGATE: define internal{{.*}} i32 @f(){{[^#]*}}#[[F_ATTRS:[0-9]+]]
// CHECK-PROPAGATE: attributes #[[F_ATTRS]] = {{.*}}"target-cpu"="skylake"

// CHECK-NO-PROPAGATE: define{{.*}} i32 @f(){{[^#]*}}#[[F_ATTRS:[0-9]+]]
// CHECK-NO-PROPAGATE-NOT: attributes #[[F_ATTRS]] = {{.*}}"target-cpu"="skylake"

#endif

// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file.bc'
