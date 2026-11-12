// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DUSE_TEMPLATE_FUNCTION=1 %s | \
// RUN: FileCheck -check-prefix=CHECK-USES-TEMPLATE-FUNCTION %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs \
// RUN: -DSPECIALIZE_TEMPLATE_FUNCTION=1 %s | \
// RUN: FileCheck -check-prefix=CHECK-SPECIALIZES-TEMPLATE-FUNCTION %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | llvm-nm - 2>&1 | count 0

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: -DUSE_TEMPLATE_FUNCTION=1 %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-USES-TEMPLATE-FUNCTION %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: -DSPECIALIZE_TEMPLATE_FUNCTION=1 %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SPECIALIZES-TEMPLATE-FUNCTION %s

// CHECK-NOT: _Z16templateFunctionIiET_S0_
// CHECK-USES-TEMPLATE-FUNCTION-DAG: _Z16templateFunctionIiET_S0_
// CHECK-SPECIALIZES-TEMPLATE-FUNCTION-DAG: _Z16templateFunctionIiET_S0_
template <typename T>
T templateFunction(T t) { return t; }

#ifdef USE_TEMPLATE_FUNCTION
int FortyTwo = templateFunction<int>(42);
#endif

#ifdef SPECIALIZE_TEMPLATE_FUNCTION
template <>
int templateFunction<int>(int t);
// TODO: Make it so that -emit-interface-stubs does not emit
// _Z16templateFunctionIiET_S0_ if there is no user of the specialization.
int foo() { return templateFunction(42); }
#endif
