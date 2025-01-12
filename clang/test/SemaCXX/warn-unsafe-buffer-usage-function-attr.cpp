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

struct HoldsUnsafeMembers {
    HoldsUnsafeMembers()
        : FromCtor(3),  // expected-warning{{function introduces unsafe buffer manipulation}}
          FromCtor2{3}  // expected-warning{{function introduces unsafe buffer manipulation}}
    {}

    [[clang::unsafe_buffer_usage]]
    HoldsUnsafeMembers(int i)
        : FromCtor(i),  // expected-warning{{function introduces unsafe buffer manipulation}}
          FromCtor2{i}  // expected-warning{{function introduces unsafe buffer manipulation}}
    {}

    HoldsUnsafeMembers(float f)
        : HoldsUnsafeMembers(0) {}  // expected-warning{{function introduces unsafe buffer manipulation}}

    UnsafeMembers FromCtor;
    UnsafeMembers FromCtor2;
    UnsafeMembers FromField{3};  // expected-warning 2{{function introduces unsafe buffer manipulation}}
};

struct SubclassUnsafeMembers : public UnsafeMembers {
    SubclassUnsafeMembers()
        : UnsafeMembers(3)  // expected-warning{{function introduces unsafe buffer manipulation}}
    {}

    [[clang::unsafe_buffer_usage]]
    SubclassUnsafeMembers(int i)
        : UnsafeMembers(i)  // expected-warning{{function introduces unsafe buffer manipulation}}
    {}
};

// https://github.com/llvm/llvm-project/issues/80482
void testClassMembers() {
    UnsafeMembers(3);  // expected-warning{{function introduces unsafe buffer manipulation}}

    (void)static_cast<int>(UnsafeMembers());  // expected-warning{{function introduces unsafe buffer manipulation}}

    UnsafeMembers().Method();  // expected-warning{{function introduces unsafe buffer manipulation}}

    UnsafeMembers()();  // expected-warning{{function introduces unsafe buffer manipulation}}

    testFoldExpression(UnsafeMembers(), UnsafeMembers());

    HoldsUnsafeMembers();
    HoldsUnsafeMembers(3);  // expected-warning{{function introduces unsafe buffer manipulation}}

    SubclassUnsafeMembers();
    SubclassUnsafeMembers(3);  // expected-warning{{function introduces unsafe buffer manipulation}}
}

// Not an aggregate, so its constructor is not implicit code and will be
// visited/checked for warnings.
struct NotCalledHoldsUnsafeMembers {
    NotCalledHoldsUnsafeMembers()
        : FromCtor(3),  // expected-warning{{function introduces unsafe buffer manipulation}}
          FromCtor2{3}  // expected-warning{{function introduces unsafe buffer manipulation}}
    {}

    UnsafeMembers FromCtor;
    UnsafeMembers FromCtor2;
    UnsafeMembers FromField{3};  // expected-warning{{function introduces unsafe buffer manipulation}}
};

// An aggregate, so its constructor is implicit code. Since it's not called, it
// is never generated.
struct AggregateUnused {
    UnsafeMembers f1;
    // While this field would trigger the warning during initialization, since
    // it's unused, there's no code generated that does the initialization, so
    // no warning.
    UnsafeMembers f2{3};
};

struct AggregateExplicitlyInitializedSafe {
    UnsafeMembers f1;
    // The warning is not fired as the field is always explicltly initialized
    // elsewhere. This initializer is never used.
    UnsafeMembers f2{3};
};

void testAggregateExplicitlyInitializedSafe() {
    AggregateExplicitlyInitializedSafe A{
        .f2 = UnsafeMembers(),  // A safe constructor.
    };
}

struct AggregateExplicitlyInitializedUnsafe {
    UnsafeMembers f1;
    // The warning is not fired as the field is always explicltly initialized
    // elsewhere. This initializer is never used.
    UnsafeMembers f2{3};
};

void testAggregateExplicitlyInitializedUnsafe() {
    AggregateExplicitlyInitializedUnsafe A{
        .f2 = UnsafeMembers(3),  // expected-warning{{function introduces unsafe buffer manipulation}}
    };
}

struct AggregateViaAggregateInit {
    UnsafeMembers f1;
    // FIXME: A construction of this class does initialize the field through
    // this initializer, so it should warn. Ideally it should also point to
    // where the site of the construction is in testAggregateViaAggregateInit().
    UnsafeMembers f2{3};
};

void testAggregateViaAggregateInit() {
    AggregateViaAggregateInit A{};
};

struct AggregateViaValueInit {
    UnsafeMembers f1;
    // FIXME: A construction of this class does initialize the field through
    // this initializer, so it should warn. Ideally it should also point to
    // where the site of the construction is in testAggregateViaValueInit().
    UnsafeMembers f2{3};
};

void testAggregateViaValueInit() {
    auto A = AggregateViaValueInit();
};

struct AggregateViaDefaultInit {
    UnsafeMembers f1;
    // FIXME: A construction of this class does initialize the field through
    // this initializer, so it should warn. Ideally it should also point to
    // where the site of the construction is in testAggregateViaValueInit().
    UnsafeMembers f2{3};
};

void testAggregateViaDefaultInit() {
    AggregateViaDefaultInit A;
};
