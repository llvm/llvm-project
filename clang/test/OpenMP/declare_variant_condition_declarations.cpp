// RUN: split-file %s %t
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -include %t/conditions.h \
// RUN:   %t/use.cpp -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -x c++-header %t/conditions.h \
// RUN:   -emit-pch -o %t/conditions.pch
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 \
// RUN:   -triple x86_64-unknown-linux -include-pch %t/conditions.pch \
// RUN:   %t/use.cpp -emit-llvm -o - | FileCheck %s

// Conditions naming different parameters remain distinct even when their
// arguments have the same value. Exercise an attribute instantiated before
// serialization as well as attributes instantiated after loading the PCH.
// CHECK-LABEL: define{{.*}} i32 @instantiated_before_pch()
// CHECK: call i32 @high()
// CHECK: ret i32
// CHECK-LABEL: define{{.*}} i32 @distinct_parameters()
// CHECK: call i32 @high()
// CHECK: call i32 @high()
// CHECK: ret i32

// Parameter names do not affect identity when their positions agree.
// CHECK-LABEL: define{{.*}} i32 @renamed_parameters()
// CHECK: call i32 @low()
// CHECK: ret i32

// Equal-valued enumerators in different scopes are different declarations.
// CHECK-LABEL: define{{.*}} i32 @different_scopes()
// CHECK: call i32 @high()
// CHECK: ret i32

// Compare a condition loaded from a PCH with one parsed in this process.
// They name the same declaration and must still form a subset relationship.
// CHECK-LABEL: define{{.*}} i32 @same_declaration()
// CHECK: call i32 @low()
// CHECK: ret i32

//--- conditions.h
extern "C" int high();
extern "C" int low();

#pragma omp declare variant(high) \
    match(user = {condition(N)}, \
          implementation = {vendor(score(100) : llvm)})
template <int N, int M> int distinct();
#pragma omp declare variant(low) \
    match(user = {condition(N)}, device = {kind(cpu)}, \
          implementation = {vendor(score(1) : llvm)})
template <int M, int N> int distinct();

extern "C" int instantiated_before_pch() { return distinct<1, 1>(); }

#pragma omp declare variant(high) \
    match(user = {condition(N)}, \
          implementation = {vendor(score(100) : llvm)})
template <int N> int renamed();
#pragma omp declare variant(low) \
    match(user = {condition(A)}, device = {kind(cpu)}, \
          implementation = {vendor(score(1) : llvm)})
template <int A> int renamed();

namespace First {
enum { A = 1 };
extern "C" {
#pragma omp declare variant(high) \
    match(user = {condition(A)}, \
          implementation = {vendor(score(100) : llvm)})
int scoped_base();
}
}

enum { Shared = 1 };
#pragma omp declare variant(high) \
    match(user = {condition(Shared)}, \
          implementation = {vendor(score(100) : llvm)})
int shared_base();

//--- use.cpp
namespace Second {
enum { A = 1 };
extern "C" {
#pragma omp declare variant(low) \
    match(user = {condition(A)}, device = {kind(cpu)}, \
          implementation = {vendor(score(1) : llvm)})
int scoped_base();
}
}

#pragma omp declare variant(low) \
    match(user = {condition(Shared)}, device = {kind(cpu)}, \
          implementation = {vendor(score(1) : llvm)})
int shared_base();

extern "C" int distinct_parameters() {
  return distinct<1, 2>() + distinct<1, 1>();
}

extern "C" int renamed_parameters() { return renamed<1>(); }

extern "C" int different_scopes() {
  return Second::scoped_base();
}

extern "C" int same_declaration() {
  return shared_base();
}
