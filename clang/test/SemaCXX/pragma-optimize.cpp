// RUN: %clang_cc1 -I %S/Inputs -x c++ -std=c++11 -triple x86_64-unknown-linux -emit-llvm -O2 < %s | FileCheck %s
// RUN: %clang_cc1 -I %S/Inputs -x c++ -std=c++11 -triple x86_64-unknown-linux -emit-llvm -Os < %s | FileCheck %s
// RUN: %clang_cc1 -I %S/Inputs -x c++ -std=c++11 -triple x86_64-unknown-linux -emit-llvm -Oz < %s | FileCheck %s

#pragma clang optimize off

// This is a macro definition and therefore its text is not present after
// preprocessing. The pragma has no effect here.
#define CREATE_FUNC(name)        \
int name (int param) {           \
    return param;                \
}                                \

// This is a declaration and therefore it is not decorated with `optnone`.
extern int foo(int a, int b);
// CHECK-DAG: @_Z3fooii{{.*}} [[ATTRFOO:#[0-9]+]]

// This is a definition and therefore it will be decorated with `optnone`.
int bar(int x, int y) {
    for(int i = 0; i < x; ++i)
        y += x;
    return y + foo(x, y);
}
// CHECK-DAG: @_Z3barii{{.*}} [[ATTRBAR:#[0-9]+]]

// The function "int created (int param)" created by the macro invocation
// is also decorated with the `optnone` attribute because it is within a
// region of code affected by the functionality (not because of the position
// of the macro definition).
CREATE_FUNC (created)
// CHECK-DAG: @_Z7createdi{{.*}} [[ATTRCREATED:#[0-9]+]]

class MyClass {
    public:
        // The declaration of the method is not decorated with `optnone`.
        int method(int blah);
};

// The definition of the method instead is decorated with `optnone`.
int MyClass::method(int blah) {
    return blah + 1;
}
// CHECK-DAG: @_ZN7MyClass6methodEi{{.*}} [[ATTRMETHOD:#[0-9]+]]

// A template declaration will not be decorated with `optnone`.
template <typename T> T twice (T param);

// The template definition will be decorated with the attribute `optnone`.
template <typename T> T thrice (T param) {
    return 3 * param;
}

// This function definition will not be decorated with `optnone` because the
// attribute would conflict with `always_inline`.
int __attribute__((always_inline)) baz(int z) {
    return foo(z, 2);
}
// CHECK-DAG: @_Z3bazi{{.*}} [[ATTRBAZ:#[0-9]+]]

// This function definition will not be decorated with `optnone` because the
// attribute would conflict with `minsize`.
int __attribute__((minsize)) bax(int z) {
    return foo(z, 2);
}
// CHECK-DAG: @_Z3baxi{{.*}} [[ATTRBAX:#[0-9]+]]

#pragma clang optimize on

// The function "int wombat(int param)" created by the macro is not
// decorated with `optnone`, because the pragma applies its effects only
// after preprocessing. The position of the macro definition is not
// relevant.
CREATE_FUNC (wombat)
// CHECK-DAG: @_Z6wombati{{.*}} [[ATTRWOMBAT:#[0-9]+]]

// This instantiation of the "twice" template function with a "float" type
// will not have an `optnone` attribute because the template declaration was
// not affected by the pragma.
float container (float par) {
    return twice(par);
}
// CHECK-DAG: @_Z9containerf{{.*}} [[ATTRCONTAINER:#[0-9]+]]
// CHECK-DAG: @_Z5twiceIfET_S0_{{.*}} [[ATTRTWICE:#[0-9]+]]

// This instantiation of the "thrice" template function with a "float" type
// will have an `optnone` attribute because the template definition was
// affected by the pragma.
float container2 (float par) {
    return thrice(par);
}
// CHECK-DAG: @_Z10container2f{{.*}} [[ATTRCONTAINER2:#[0-9]+]]
// CHECK-DAG: @_Z6thriceIfET_S0_{{.*}} [[ATTRTHRICEFLOAT:#[0-9]+]]


// A template specialization is a new definition and it will not be
// decorated with an `optnone` attribute because it is now outside of the
// affected region.
template<> int thrice(int par) {
    return (par << 1) + par;
}
int container3 (int par) {
    return thrice(par);
}
// CHECK-DAG: @_Z10container3i{{.*}} [[ATTRCONTAINER3:#[0-9]+]]
// CHECK-DAG: @_Z6thriceIiET_S0_{{.*}} [[ATTRTHRICEINT:#[0-9]+]]


// Test that we can re-open and re-close an "off" region after the first one,
// and that this works as expected.

#pragma clang optimize off

int another_optnone(int x) {
    return x << 1;
}
// CHECK-DAG: @_Z15another_optnonei{{.*}} [[ATTRANOTHEROPTNONE:#[0-9]+]]

#pragma clang optimize on

int another_normal(int x) {
    return x << 2;
}
// CHECK-DAG: @_Z14another_normali{{.*}} [[ATTRANOTHERNORMAL:#[0-9]+]]


// Test that we can re-open an "off" region by including a header with the
// pragma and that this works as expected (i.e. the off region "falls through"
// the end of the header into this file).

#include <header-with-pragma-optimize-off.h>

int yet_another_optnone(int x) {
    return x << 3;
}
// CHECK-DAG: @_Z19yet_another_optnonei{{.*}} [[ATTRYETANOTHEROPTNONE:#[0-9]+]]

#pragma clang optimize on

int yet_another_normal(int x) {
    return x << 4;
}
// CHECK-DAG: @_Z18yet_another_normali{{.*}} [[ATTRYETANOTHERNORMAL:#[0-9]+]]


// Check for both noinline and optnone on each function that should have them.
// CHECK-DAG: attributes [[ATTRBAR]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes [[ATTRCREATED]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes [[ATTRMETHOD]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes [[ATTRTHRICEFLOAT]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes [[ATTRANOTHEROPTNONE]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes [[ATTRYETANOTHEROPTNONE]] = { {{.*}}noinline{{.*}}optnone{{.*}} }

// Check that the other functions do NOT have optnone.
// CHECK-NOT: attributes [[ATTRFOO]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRBAZ]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRBAX]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRWOMBAT]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRCONTAINER]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRTWICE]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRCONTAINER2]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRCONTAINER3]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRTHRICEINT]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRANOTHERNORMAL]] = { {{.*}}optnone{{.*}} }
// CHECK-NOT: attributes [[ATTRYETANOTHERNORMAL]] = { {{.*}}optnone{{.*}} }
