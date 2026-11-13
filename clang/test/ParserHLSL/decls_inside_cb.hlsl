// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -DCBUFFER -o - -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -o - -fsyntax-only %s -verify

// expected-no-diagnostics

namespace Ns1 {
    class Foo {
        int a;
    };

    namespace Ns2 {
        class Bar {
            int b;
        };
    }
}

#ifdef CBUFFER
cbuffer A {
#else
tbuffer A {
#endif
    // variable declaration
    float f;

    // class declaration
    class MyClass;
    class MyClass {
        float f;
    };

    // struct declaration
    struct MyStruct {
        float f;
    };
    class MyStruct;

    // function declaration
    float getF();
    float getF() {
        return f;
    }

    // empty declaration
    ;

    // enumeration declaration
    enum MyEnum {
        Value1,
        Value2
    };

    // template function declaration
    template<typename T> T func(T t) { return t; }

    // template struct declaration
    template<typename T> struct S { float s; };

    // type alias declaration
    using F32 = float;

    // template type alias declaration
    template<typename T> using A = S<T>;

    // using namespace declaration
    using namespace Ns1;

    Foo foo;

    // using declaration
    using Ns2::Bar;

    Bar bar;

    // typedef declaration
    typedef float F32Typedef;

    // export declarations
    export {
        void externalFn1();
    }

    export void externalFn2();

    // static assert declarations
    static_assert(sizeof(float) == 4, "something's wrong");
}
