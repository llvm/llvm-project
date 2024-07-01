// RUN: %clang_cc1 -triple x86_64-linux -verify -emit-llvm-only %s

// Reject this function; there's no way to correctly destroy the temporary
struct S { S(); ~S(); };
struct S2 { S s1, s2; };
S2 f() {
    return {S(), ({
        while (true) {
            return {S(), ({break; S();})}; // expected-error {{cannot compile this nested return statement yet}}
        }
        S();})};
}

// This variant doesn't have any temporaries, so it's allowed.
struct Simple { int s1, s2; };
Simple f2() {
    return {1, ({
        while (true) {
            return {2, ({break; 3;})};
        }
        3;})};
}
