/*
 * Compiler Builtins Stubs for Kernel Module
 *
 * These stub implementations provide the symbols that rustc's compiler-builtins
 * may reference but which should never actually be called in kernel code.
 * The f128 operations are not used in our DSMIL code, but the compiler
 * includes them as weak references.
 */

typedef struct {
    unsigned long long low;
    unsigned long long high;
} float128_t;

/* Quadruple-precision (f128) arithmetic stubs */
float128_t __multf3(float128_t a, float128_t b) {
    float128_t result = {0, 0};
    return result;
}

float128_t __addtf3(float128_t a, float128_t b) {
    float128_t result = {0, 0};
    return result;
}

float128_t __subtf3(float128_t a, float128_t b) {
    float128_t result = {0, 0};
    return result;
}

float128_t __divtf3(float128_t a, float128_t b) {
    float128_t result = {0, 0};
    return result;
}

int __eqtf2(float128_t a, float128_t b) {
    return 0;
}

int __netf2(float128_t a, float128_t b) {
    return 0;
}

int __getf2(float128_t a, float128_t b) {
    return 0;
}

int __gttf2(float128_t a, float128_t b) {
    return 0;
}

int __letf2(float128_t a, float128_t b) {
    return 0;
}

int __lttf2(float128_t a, float128_t b) {
    return 0;
}

int __unordtf2(float128_t a, float128_t b) {
    return 0;
}

float128_t __floatsitf(int i) {
    float128_t result = {0, 0};
    return result;
}

float128_t __floatditf(long long i) {
    float128_t result = {0, 0};
    return result;
}

float128_t __floattitf(__int128 i) {
    float128_t result = {0, 0};
    return result;
}

float128_t __floatunsitf(unsigned int i) {
    float128_t result = {0, 0};
    return result;
}

float128_t __floatunditf(unsigned long long i) {
    float128_t result = {0, 0};
    return result;
}

float128_t __floatuntitf(unsigned __int128 i) {
    float128_t result = {0, 0};
    return result;
}

int __fixtfsi(float128_t a) {
    return 0;
}

long long __fixtfdi(float128_t a) {
    return 0;
}

__int128 __fixtfti(float128_t a) {
    return 0;
}

unsigned int __fixunstfsi(float128_t a) {
    return 0;
}

unsigned long long __fixunstfdi(float128_t a) {
    return 0;
}

unsigned __int128 __fixunstfti(float128_t a) {
    return 0;
}

float128_t __extendsftf(float f) {
    float128_t result = {0, 0};
    return result;
}

float128_t __extenddftf(double d) {
    float128_t result = {0, 0};
    return result;
}

float __trunctfsf(float128_t a) {
    return 0.0f;
}

double __trunctfdf(float128_t a) {
    return 0.0;
}
