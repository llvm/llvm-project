// RUN: %clang_cc1 %s -debug-info-kind=limited -triple %itanium_abi_triple -fno-use-cxa-atexit -S -disable-O0-optnone  -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-NOKEXT
// RUN: %clang_cc1 %s -debug-info-kind=limited -triple %itanium_abi_triple -fno-use-cxa-atexit -fapple-kext -S -disable-O0-optnone -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-KEXT

class A {
 public:
  A() {}
  virtual ~A() {}
};

A glob;
A array[2];

void foo() {
  static A stat;
}

// CHECK-NOKEXT: !DISubprogram(name: "__cxx_global_var_init",{{.*}} line: 12,{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__dtor_glob",{{.*}} line: 12,{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__cxx_global_var_init.1",{{.*}} line: 13,{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__cxx_global_array_dtor",{{.*}} line: 13,{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__dtor_array",{{.*}} line: 13,{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__dtor__ZZ3foovE4stat",{{.*}} line: 16,{{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram({{.*}} DISPFlagLocalToUnit | DISPFlagDefinition

// CHECK-KEXT: !DISubprogram({{.*}} DISPFlagLocalToUnit | DISPFlagDefinition
