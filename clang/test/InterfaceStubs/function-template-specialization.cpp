// REQUIRES: x86-registered-target

// TODO: Fix the case in llvm-ifs where it crashes on an empty Symbols list.
// RUN: %clang -target x86_64-unknown-linux-gnu -o - -emit-interface-stubs -c \
// RUN: -interface-stub-version=experimental-ifs-v1 %s | FileCheck %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -emit-interface-stubs -emit-merged-ifs \
// RUN: -interface-stub-version=experimental-ifs-v1 \
// RUN: -DUSE_TEMPLATE_FUNCTION=1 %s | \
// RUN: FileCheck -check-prefix=CHECK-USES-TEMPLATE-FUNCTION %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -emit-interface-stubs -emit-merged-ifs \
// RUN: -interface-stub-version=experimental-ifs-v1 \
// RUN: -DSPECIALIZE_TEMPLATE_FUNCTION=1 %s | \
// RUN: FileCheck -check-prefix=CHECK-SPECIALIZES-TEMPLATE-FUNCTION %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c \
// RUN: %s | llvm-nm - 2>&1 | FileCheck %s

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
