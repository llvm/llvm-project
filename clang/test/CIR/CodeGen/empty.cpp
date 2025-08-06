// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// These declarations shouldn't emit any code. Therefore the module is expected to be empty.

template<typename T>
concept some_concept = true;

template<some_concept T>
class class_template {};

; // Empty declaration

template<typename T>
void function_template();

static_assert(true, "top level static assert");

template<typename T>
using type_alias = T;

namespace N {
    using ::class_template; // UsingShadow
}

template<typename T>
struct deduction_guide {};

deduction_guide() -> deduction_guide<int>;

// CIR: module {{.*}} {
// CIR-NEXT: }
