// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Operator overloads are missing comdat groups.
//
// CodeGen generates comdat for inline operator overloads:
//   $_ZN3IntplERKS_ = comdat any
//   define linkonce_odr ... @_ZN3IntplERKS_(...) comdat
//
// CIR omits comdat:
//   define linkonce_odr ... @_ZN3IntplERKS_(...)  // No comdat
//
// This affects all inline operator overloads:
// - Arithmetic operators (+, -, *, /, etc.)
// - Comparison operators (==, !=, <, >, etc.)
// - Assignment operators (=, +=, -=, etc.)
// - Subscript operator []
// - Call operator ()
// - Conversion operators
//
// Impact: Potential ODR violations in multi-TU programs

// DIFF: -$_ZN3IntplERKS_ = comdat any
// DIFF: -define linkonce_odr {{.*}} @_ZN3IntplERKS_{{.*}} comdat
// DIFF: +define linkonce_odr {{.*}} @_ZN3IntplERKS_

// Arithmetic operator
struct Int {
    int value;

    Int operator+(const Int& other) const {
        return {value + other.value};
    }

    Int operator-(const Int& other) const {
        return {value - other.value};
    }
};

int test_arithmetic() {
    Int a{10}, b{20};
    Int c = a + b;
    return c.value;
}

// Comparison operators
struct Comparable {
    int value;

    bool operator==(const Comparable& other) const {
        return value == other.value;
    }

    bool operator<(const Comparable& other) const {
        return value < other.value;
    }
};

int test_comparison() {
    Comparable a{10}, b{20};
    return a < b ? 1 : 0;
}

// Assignment operator
struct Assignable {
    int value;

    Assignable& operator=(const Assignable& other) {
        value = other.value;
        return *this;
    }
};

int test_assignment() {
    Assignable a{10}, b{20};
    a = b;
    return a.value;
}

// Subscript operator
struct Array {
    int data[3] = {1, 2, 3};

    int& operator[](int index) {
        return data[index];
    }

    const int& operator[](int index) const {
        return data[index];
    }
};

int test_subscript() {
    Array arr;
    return arr[1];
}

// Call operator (functor)
struct Adder {
    int operator()(int a, int b) const {
        return a + b;
    }
};

int test_call() {
    Adder add;
    return add(10, 20);
}

// Conversion operator
struct Convertible {
    int value;

    operator int() const {
        return value;
    }

    operator bool() const {
        return value != 0;
    }
};

int test_conversion() {
    Convertible c{42};
    int x = c;  // Uses conversion operator
    return x;
}
