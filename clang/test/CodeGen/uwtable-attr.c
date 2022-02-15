// Test that function and modules attributes react on the command-line options,
// it does not state the current behaviour makes sense in all cases (it does not).

// RUN: %clang -target x86_64-linux -S -emit-llvm -o - %s                                                    | FileCheck %s -check-prefixes=CHECK,DEFAULT
// RUN: %clang -target x86_64-linux -S -emit-llvm -o - %s -funwind-tables    -fno-asynchronous-unwind-tables | FileCheck %s -check-prefixes=CHECK,TABLES
// RUN: %clang -target x86_64-linux -S -emit-llvm -o - %s -fno-unwind-tables -fno-asynchronous-unwind-tables | FileCheck %s -check-prefixes=CHECK,NO_TABLES

// RUN: %clang -target x86_64-linux -S -emit-llvm -o - -x c++ %s                                                                     | FileCheck %s -check-prefixes=CHECK,DEFAULT
// RUN: %clang -target x86_64-linux -S -emit-llvm -o - -x c++ %s                  -funwind-tables    -fno-asynchronous-unwind-tables | FileCheck %s -check-prefixes=CHECK,TABLES
// RUN: %clang -target x86_64-linux -S -emit-llvm -o - -x c++ %s  -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables | FileCheck %s -check-prefixes=CHECK,NO_TABLES

// REQUIRES: x86-registered-target

#ifdef __cplusplus
extern "C" void g(void);
struct S { ~S(); };
extern "C" int f() { S s; g(); return 0;};
#else
void g(void);
int f() { g(); return 0; };
#endif

// CHECK: define {{.*}} @f() #[[#F:]]
// CHECK: declare {{.*}} @g() #[[#]]

// DEFAULT: attributes #[[#F]] = { {{.*}} uwtable{{ }}{{.*}} }
// DEFAULT: ![[#]] = !{i32 7, !"uwtable", i32 2}

// TABLES: attributes #[[#F]] = { {{.*}} uwtable(sync){{.*}} }
// TABLES: ![[#]] = !{i32 7, !"uwtable", i32 1}

// NO_TABLES-NOT: uwtable
