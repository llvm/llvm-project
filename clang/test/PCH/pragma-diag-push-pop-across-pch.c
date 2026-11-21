// Test that #pragma diagnostic push in a PCH is matched by pop in the main file.

// RUN: %clang_cc1 %S/Inputs/pragma-diag-push.h -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -verify -fsyntax-only
// expected-no-diagnostics

#pragma clang diagnostic pop

int main(void) {
    return 0;
}
