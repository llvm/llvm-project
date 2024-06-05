// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions -verify %s

[[clang::unsafe_buffer_usage]]
void deprecatedFunction3();

void deprecatedFunction4(int z);

void someFunction();

[[clang::unsafe_buffer_usage]]
void overloading(int* x);

void overloading(char c[]);

void overloading(int* x, int size);

[[clang::unsafe_buffer_usage]]
void deprecatedFunction4(int z);

void caller(int z, int* x, int size, char c[]) {
    deprecatedFunction3(); // expected-warning{{function introduces unsafe buffer manipulation}}
    deprecatedFunction4(z); // expected-warning{{function introduces unsafe buffer manipulation}}
    someFunction();

    overloading(x); // expected-warning{{function introduces unsafe buffer manipulation}}
    overloading(x, size);
    overloading(c);
}

[[clang::unsafe_buffer_usage]]
void overloading(char c[]);

// Test variadics
[[clang::unsafe_buffer_usage]]
void testVariadics(int *ptr, ...);

template<typename T, typename... Args>
[[clang::unsafe_buffer_usage]]
T adder(T first, Args... args);

template <typename T>
void foo(T t) {}

template<>
[[clang::unsafe_buffer_usage]]
void foo<int *>(int *t) {}

void caller1(int *p, int *q) {
    testVariadics(p, q);  // expected-warning{{function introduces unsafe buffer manipulation}}
    adder(p, q);  // expected-warning{{function introduces unsafe buffer manipulation}}
    
    int x;
    foo(x);
    foo(&x);  // expected-warning{{function introduces unsafe buffer manipulation}}
}

// Test virtual functions
class BaseClass {
public:
    [[clang::unsafe_buffer_usage]]
    virtual void func() {}
    
    virtual void func1() {}
};

class DerivedClass : public BaseClass {
public:
    void func() {}
    
    [[clang::unsafe_buffer_usage]]
    void func1() {}
};

void testInheritance() {
    DerivedClass DC;
    DC.func();
    DC.func1();  // expected-warning{{function introduces unsafe buffer manipulation}}
    
    BaseClass *BC;
    BC->func();  // expected-warning{{function introduces unsafe buffer manipulation}}
    BC->func1();
    
    BC = &DC;
    BC->func();  // expected-warning{{function introduces unsafe buffer manipulation}}
    BC->func1();
}

class UnsafeMembers {
public:
    UnsafeMembers() {}

    [[clang::unsafe_buffer_usage]]
    UnsafeMembers(int) {}

    [[clang::unsafe_buffer_usage]]
    explicit operator int() { return 0; }

    [[clang::unsafe_buffer_usage]]
    void Method() {}

    [[clang::unsafe_buffer_usage]]
    void operator()() {}

    [[clang::unsafe_buffer_usage]]
    int operator+(UnsafeMembers) { return 0; }
};

template <class... Vs>
int testFoldExpression(Vs&&... v) {
    return (... + v);  // expected-warning{{function introduces unsafe buffer manipulation}}
}

// https://github.com/llvm/llvm-project/issues/80482
void testClassMembers() {
    UnsafeMembers(3);  // expected-warning{{function introduces unsafe buffer manipulation}}

    (void)static_cast<int>(UnsafeMembers());  // expected-warning{{function introduces unsafe buffer manipulation}}

    UnsafeMembers().Method();  // expected-warning{{function introduces unsafe buffer manipulation}}

    UnsafeMembers()();  // expected-warning{{function introduces unsafe buffer manipulation}}

    testFoldExpression(UnsafeMembers(), UnsafeMembers());
}
