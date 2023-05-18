// RUN: %clang_cc1 %s -debug-info-kind=limited -triple %itanium_abi_triple -fno-use-cxa-atexit -S -disable-O0-optnone  -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-NOKEXT
// RUN: %clang_cc1 %s -debug-info-kind=limited -triple %itanium_abi_triple -fno-use-cxa-atexit -fapple-kext -S -disable-O0-optnone -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-KEXT
// RUN: %clang_cc1 %s -gcodeview -debug-info-kind=limited -triple x86_64-windows-msvc -fno-use-cxa-atexit -S -disable-O0-optnone  -emit-llvm -o - \
// RUN:     | FileCheck %s --check-prefix=CHECK-MSVC

class A {
public:
  A();
  A(int x);
  virtual ~A();
};

A glob;
A array[2];

void foo() {
  static A stat;
}

template <typename T>
struct FooTpl {
  template <typename U>
  static A sdm_tpl;
};
template <typename T>
template <typename U>
A FooTpl<T>::sdm_tpl(sizeof(U) + sizeof(T));
template A FooTpl<int>::sdm_tpl<int>;

// CHECK-NOKEXT: !DISubprogram(name: "__cxx_global_var_init",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__dtor_glob",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__cxx_global_var_init.1",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__cxx_global_array_dtor",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__dtor_array",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram(name: "__dtor__ZZ3foovE4stat",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-NOKEXT: !DISubprogram({{.*}} DISPFlagLocalToUnit | DISPFlagDefinition

// CHECK-KEXT: !DISubprogram({{.*}} DISPFlagLocalToUnit | DISPFlagDefinition

// CHECK-MSVC: !DISubprogram(name: "`dynamic initializer for 'glob'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-MSVC: !DISubprogram(name: "`dynamic atexit destructor for 'glob'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-MSVC: !DISubprogram(name: "`dynamic initializer for 'array'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-MSVC: !DISubprogram(name: "__cxx_global_array_dtor",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-MSVC: !DISubprogram(name: "`dynamic atexit destructor for 'array'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-MSVC: !DISubprogram(name: "`dynamic atexit destructor for 'stat'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition

// MSVC does weird stuff when templates are involved, so we don't match exactly,
// but these names are reasonable.
// FIXME: These should not be marked DISPFlagLocalToUnit.
// CHECK-MSVC: !DISubprogram(name: "FooTpl<int>::`dynamic initializer for 'sdm_tpl<int>'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-MSVC: !DISubprogram(name: "FooTpl<int>::`dynamic atexit destructor for 'sdm_tpl<int>'",{{.*}} flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
