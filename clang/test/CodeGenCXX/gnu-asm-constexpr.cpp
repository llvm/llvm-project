// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64 %s -S -o /dev/null -Werror -verify
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

struct string_view {
    int S;
    const char* D;
    constexpr string_view() : S(0), D(0){}
    constexpr string_view(const char* Str) : S(__builtin_strlen(Str)), D(Str) {}
    constexpr string_view(int Size, const char* Str) : S(Size), D(Str) {}
    constexpr int size() const {
        return S;
    }
    constexpr const char* data() const {
        return D;
    }
};

namespace GH143242 {
    constexpr string_view code2 = R"(nop; nop; nop; nop)";
    asm((code2));
    // CHECK: module asm "nop; nop; nop; nop"
}

int func() {return 0;};

void f() {

    asm((string_view("")) ::(string_view("r"))(func()));
    // CHECK: %[[CALL:.*]] = call noundef i32 @_Z4funcv
    // CHECK: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"
    asm("" :::(string_view("memory")));
    // CHECK: call void asm sideeffect "", "~{memory},~{dirflag},~{fpsr},~{flags}"
}

void foo(unsigned long long addr, unsigned long long a0) {
    register unsigned long long result asm("rax");
    register unsigned long long b0 asm("rdi");

    b0 = a0;

    asm((string_view("call *%1")) : (string_view("=r")) (result)
        : (string_view("r"))(addr), (string_view("r")) (b0) : (string_view("memory")));

    // CHECK:{{.*}} call i64 asm "call *$1", "={rax},r,{rdi},~{memory},~{dirflag},~{fpsr},~{flags}"
}


void test_srcloc() {
    asm((string_view( // expected-error {{invalid instruction mnemonic 'nonsense'}} \
                      // expected-error {{invalid instruction mnemonic 'foobar'}} \
                      // expected-note@1 {{instantiated into assembly here}} \
                      // expected-note@2 {{instantiated into assembly here}}
        R"o(nonsense
        foobar)o")

    ) ::(string_view("r"))(func()));
}
