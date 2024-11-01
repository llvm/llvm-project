// RUN: %clang_cc1 -std=c++20 -emit-obj -verify -o %t.pcm %s

export module M;
export { // expected-note 2{{export block begins here}}
    union { bool a; }; // expected-error {{anonymous unions at namespace or global scope must be declared 'static'}} expected-error {{declaration of 'a' with internal linkage cannot be exported}}
    static union { bool a; }; // expected-error {{declaration of 'a' with internal linkage cannot be exported}}
}
