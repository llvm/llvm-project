// RUN: %clang_cc1 -S -O3 -fno-builtin -o - %s | FileCheck %s
// RUN: %clang_cc1 -S -O3 -fno-builtin-ceil -fno-builtin-copysign -fno-builtin-cos \
// RUN:  -fno-builtin-fabs -fno-builtin-floor -fno-builtin-strcat -fno-builtin-strncat \
// RUN:  -fno-builtin-strchr -fno-builtin-strrchr -fno-builtin-strcmp -fno-builtin-strncmp \
// RUN:  -fno-builtin-strcpy -fno-builtin-stpcpy -fno-builtin-strncpy -fno-builtin-strlen \
// RUN:  -fno-builtin-strpbrk -fno-builtin-strspn -fno-builtin-strtod -fno-builtin-strtof \
// RUN:  -fno-builtin-strtold -fno-builtin-strtol -fno-builtin-strtoll -fno-builtin-strtoul \
// RUN:  -fno-builtin-strtoull -o - %s | FileCheck %s
// rdar://10551066

typedef __SIZE_TYPE__ size_t;

double ceil(double x);
double copysign(double,double);
double cos(double x);
double fabs(double x);
double floor(double x);
char *strcat(char *s1, const char *s2);
char *strncat(char *s1, const char *s2, size_t n);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
int strcmp(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
char *strcpy(char *s1, const char *s2);
char *stpcpy(char *s1, const char *s2);
char *strncpy(char *s1, const char *s2, size_t n);
size_t strlen(const char *s);
char *strpbrk(const char *s1, const char *s2);
size_t strspn(const char *s1, const char *s2);
double strtod(const char *nptr, char **endptr);
float strtof(const char *nptr, char **endptr);
long double strtold(const char *nptr, char **endptr);
long int strtol(const char *nptr, char **endptr, int base);
long long int strtoll(const char *nptr, char **endptr, int base);
unsigned long int strtoul(const char *nptr, char **endptr, int base);
unsigned long long int strtoull(const char *nptr, char **endptr, int base);

double t1(double x) { return ceil(x); }
// CHECK: t1
// CHECK: ceil

double t2(double x, double y) { return copysign(x,y); }
// CHECK: t2
// CHECK: copysign

double t3(double x) { return cos(x); }
// CHECK: t3
// CHECK: cos

double t4(double x) { return fabs(x); }
// CHECK: t4
// CHECK: fabs

double t5(double x) { return floor(x); }
// CHECK: t5
// CHECK: floor

char *t6(char *x) { return strcat(x, ""); }
// CHECK: t6
// CHECK: strcat

char *t7(char *x) { return strncat(x, "", 1); }
// CHECK: t7
// CHECK: strncat

char *t8(void) { return strchr("hello, world", 'w'); }
// CHECK: t8
// CHECK: strchr

char *t9(void) { return strrchr("hello, world", 'w'); }
// CHECK: t9
// CHECK: strrchr

int t10(void) { return strcmp("foo", "bar"); }
// CHECK: t10
// CHECK: strcmp

int t11(void) { return strncmp("foo", "bar", 3); }
// CHECK: t11
// CHECK: strncmp

char *t12(char *x) { return strcpy(x, "foo"); }
// CHECK: t12
// CHECK: strcpy

char *t13(char *x) { return stpcpy(x, "foo"); }
// CHECK: t13
// CHECK: stpcpy

char *t14(char *x) { return strncpy(x, "foo", 3); }
// CHECK: t14
// CHECK: strncpy

size_t t15(void) { return strlen("foo"); }
// CHECK: t15
// CHECK: strlen

char *t16(char *x) { return strpbrk(x, ""); }
// CHECK: t16
// CHECK: strpbrk

size_t t17(char *x) { return strspn(x, ""); }
// CHECK: t17
// CHECK: strspn

double t18(char **x) { return strtod("123.4", x); }
// CHECK: t18
// CHECK: strtod

float t19(char **x) { return strtof("123.4", x); }
// CHECK: t19
// CHECK: strtof

long double t20(char **x) { return strtold("123.4", x); }
// CHECK: t20
// CHECK: strtold

long int t21(char **x) { return strtol("1234", x, 10); }
// CHECK: t21
// CHECK: strtol

long int t22(char **x) { return strtoll("1234", x, 10); }
// CHECK: t22
// CHECK: strtoll

long int t23(char **x) { return strtoul("1234", x, 10); }
// CHECK: t23
// CHECK: strtoul

long int t24(char **x) { return strtoull("1234", x, 10); }
// CHECK: t24
// CHECK: strtoull
