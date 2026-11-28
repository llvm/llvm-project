// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -Wno-unused-value -x c++ -triple x86_64-pc-windows-msvc -emit-llvm %s -o - | FileCheck %s --check-prefix=MS
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -Wno-unused-value -x c++ -triple x86_64-pc-windows-msvc -fcxx-exceptions -fexceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=MS
// RUN: %clang_cc1 -verify -fopenmp -Wno-openmp-mapping -Wno-unused-value -x c++ -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=ITANIUM
// expected-no-diagnostics

// Mapping a dynamic class emits its vtable for offloading. The MS ABI emits
// the deleting destructor with the vtable, so Sema has to prepare it even
// when the destructor is only declared here. This used to crash.

class Implicit { public: virtual ~Implicit(); };
class Explicit { public: virtual ~Explicit(); };
class Pointee { public: virtual ~Pointee(); };
class Member { public: virtual ~Member(); };
class Holder { public: virtual ~Holder(); Member M; };
class Base { public: virtual ~Base(); };
class Derived : public Base { public: ~Derived() override; };

extern Implicit I;
extern Explicit E;
extern Pointee *P;
extern Holder H;
extern Derived D;

// MS-DAG: @"??_7Implicit@@6B@" = alias
// MS-DAG: @"??_7Explicit@@6B@" = alias
// MS-DAG: @"??_7Pointee@@6B@" = alias
// MS-DAG: @"??_7Holder@@6B@" = alias
// MS-DAG: @"??_7Member@@6B@" = alias
// MS-DAG: @"??_7Derived@@6B@" = alias
// MS-DAG: @"??_7Base@@6B@" = alias
// MS-DAG: define {{.*}}@"??_GImplicit@@{{[^"]*}}"(
// MS-DAG: define {{.*}}@"??_GExplicit@@{{[^"]*}}"(
// MS-DAG: define {{.*}}@"??_GPointee@@{{[^"]*}}"(
// MS-DAG: define {{.*}}@"??_GHolder@@{{[^"]*}}"(
// MS-DAG: define {{.*}}@"??_GMember@@{{[^"]*}}"(
// MS-DAG: define {{.*}}@"??_GDerived@@{{[^"]*}}"(
// MS-DAG: define {{.*}}@"??_GBase@@{{[^"]*}}"(

// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV8Implicit_l{{[0-9]+}}addr =
// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV8Explicit_l{{[0-9]+}}addr =
// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV7Pointee_l{{[0-9]+}}addr =
// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV6Holder_l{{[0-9]+}}addr =
// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV6Member_l{{[0-9]+}}addr =
// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV7Derived_l{{[0-9]+}}addr =
// ITANIUM-DAG: @.offloading.entry.{{.*}}_ZTV4Base_l{{[0-9]+}}addr =
void test() {
#pragma omp target
  I;
#pragma omp target map(E)
  E;
#pragma omp target map(P)
  P;
#pragma omp target map(H)
  H;
#pragma omp target map(D)
  D;
}
