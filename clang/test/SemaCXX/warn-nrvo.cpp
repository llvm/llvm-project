// RUN: %clang_cc1 -fsyntax-only -Wnrvo -verify %s
struct MyClass {
    int value;
    int c;
    MyClass(int v) : value(v), c(0) {}
    MyClass(const MyClass& other) : value(other.value) { c++; }
};

MyClass create_object(bool condition) {
    MyClass obj1(1);
    MyClass obj2(2);
    if (condition) {
        return obj1; // expected-warning{{not eliding copy on return}}
    }
    return obj2; // expected-warning{{not eliding copy on return}}
}
