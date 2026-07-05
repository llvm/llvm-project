// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-linux -verify=no-diag \
// RUN: -DDIV_CC -DDIV_RC -DDIVASSIGN -DDIVMIXEDFD -DDIVMIXEDFD2 -DDIVMIXEDID -DDIVASSIGN_MIXEDFD

// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify=no-diag
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIV_CC
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIV_RC
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIVASSIGN
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIVMIXEDFD
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIVMIXEDFD2
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIVMIXEDID
// RUN: %clang_cc1 %s -complex-range=promoted -fsyntax-only -triple x86_64-unknown-windows -verify -DDIVASSIGN_MIXEDFD

_Complex double div_ccf(_Complex float a, _Complex float b) {
    return a / b;
}

_Complex double div_cr(_Complex double a, double b) {
    return a / b;
}

_Complex double div_cr_mixed1(_Complex double a, float b) {
    return a / b;
}

_Complex double div_cr_mixed2(_Complex float a, double b) {
    return a / b;
}

_Complex double div_rr(double a, double b) {
    return a / b;
}

_Complex int div_ii(_Complex int a, _Complex int b) {
    return a / b;
}

struct UserT {
    friend UserT operator/(UserT, _Complex double);
    friend UserT operator/(_Complex double, UserT);
};

UserT div_uc(UserT a, _Complex double b) {
    return a / b;
}

UserT div_cu(_Complex double a, UserT b) {
    return a / b;
}

#ifdef DIV_CC
_Complex double div_cc(_Complex double a, const _Complex double b) {
    return a / b; // #1
}
#endif // DIV_CC

#ifdef DIV_RC
_Complex double div_rc(double a, _Complex float b) {
    return a / b; // #1
}
#endif // DIV_RC

#ifdef DIVASSIGN
_Complex double divassign(_Complex double a, _Complex double b) {
    return a /= b; // #1
}
#endif // DIVASSIGN

#ifdef DIVMIXEDFD
_Complex double divmixedfd(_Complex float a, _Complex double b) {
    return a / b; // #1
}
#endif // DIVMIXEDFD

#ifdef DIVMIXEDFD2
_Complex double divmixedfd2(_Complex double a, _Complex float b) {
    return a / b; // #1
}
#endif // DIVMIXEDFD2

#ifdef DIVMIXEDID
_Complex double divmixedid(_Complex int a, _Complex double b) {
    return a / b; // #1
}
#endif // DIVMIXEDID

#ifdef DIVASSIGN_MIXEDFD
_Complex double divassign_mixedfd(_Complex float a, _Complex double b) {
    return a /= b; // #1
}
#endif // DIVMIXEDFD

// no-diag-no-diagnostics
// expected-warning@#1 {{excess precision is requested but the target does not support excess precision which may result in observable differences in complex division behavior}}
