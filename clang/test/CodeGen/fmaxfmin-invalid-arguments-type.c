// RUN: not %clang_cc1 -triple x86_64 %s -fsyntax-only -verify 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

float fminimum_numf (float, float);
double fminimum_num (double, double);
long double fminimum_numl (long double, long double);
float fmaximum_numf (float, float);
double fmaximum_num (double, double);
long double fmaximum_numl (long double, long double);

// CHECK-ERR: passing 'char *' to parameter of incompatible type 'float'
float fmin1(char *a, char *b) {
        return fminimum_numf(a, b);
}
// CHECK-ERR: passing 'char *' to parameter of incompatible type 'double'
float fmin2(char *a, char *b) {
        return fminimum_num(a, b);
}
// CHECK-ERR: passing 'char *' to parameter of incompatible type 'long double'
float fmin3(char *a, char *b) {
        return fminimum_numl(a, b);
}
// CHECK-ERR: passing 'char *' to parameter of incompatible type 'float'
float fmax1(char *a, char *b) {
        return fmaximum_numf(a, b);
}
// CHECK-ERR: passing 'char *' to parameter of incompatible type 'double'
float fmax2(char *a, char *b) {
        return fmaximum_num(a, b);
}
// CHECK-ERR: passing 'char *' to parameter of incompatible type 'long double'
float fmax3(char *a, char *b) {
        return fmaximum_numl(a, b);
}
