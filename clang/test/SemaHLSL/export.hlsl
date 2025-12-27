// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - %s -verify

export void f1();

export void f1() {}

namespace { // expected-note {{anonymous namespace begins here}}
    export void f2(); // expected-error {{export declaration appears within anonymous namespace}}
}

export void f3();

export { // expected-note {{export block begins here}}
    void f4() {}
    export void f5() {} // expected-error {{export declaration appears within another export declaration}}
    int A; // expected-error {{export declaration can only be used on functions}}
    namespace ns { // expected-error {{export declaration can only be used on functions}}
        void f6();
    }
}

export { // expected-note {{export block begins here}}
    export { // expected-error {{export declaration appears within another export declaration}}
        void f();
    }
}

void export f7() {} // expected-error {{expected unqualified-id}}

export static void f8() {} // expected-error {{declaration of 'f8' with internal linkage cannot be exported}}

export void f9(); // expected-note {{previous declaration is here}}
static void f9(); // expected-error {{static declaration of 'f9' follows non-static declaration}}

static void f10(); // expected-note {{previous declaration is here}}
export void f10(); // expected-error {{cannot export redeclaration 'f10' here since the previous declaration has internal linkage}}

export void f11();
void f11() {}

void f12();           //   expected-note{{previous declaration is here}}
export void f12() {}  //   expected-error{{cannot export redeclaration 'f12' here since the previous declaration is not exported}}

export float V1; // expected-error {{export declaration can only be used on functions}}

static export float V2; // expected-error{{expected unqualified-id}}

export static float V3 = 0; // expected-error {{export declaration can only be used on functions}}

export groupshared float V4; // expected-error {{export declaration can only be used on functions}}

void f6() {
  export int i;  // expected-error {{expected expression}}
}

export cbuffer CB { // expected-error {{export declaration can only be used on functions}}
    int a;
}

export template<typename T> void tf1(T t) {} // expected-error {{export declaration can only be used on functions}}

void f5() export {} // expected-error {{expected function body after function declarator}}
