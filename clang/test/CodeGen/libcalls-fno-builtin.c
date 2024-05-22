// RUN: %clang_cc1 -S -emit-llvm -fno-builtin -o - %s | FileCheck --check-prefixes=GLOBAL,CHECK %s
// RUN: %clang_cc1 -S -emit-llvm -fno-builtin-ceil -fno-builtin-copysign -fno-builtin-cos \
// RUN:  -fno-builtin-fabs -fno-builtin-floor -fno-builtin-strcat -fno-builtin-strncat \
// RUN:  -fno-builtin-strchr -fno-builtin-strrchr -fno-builtin-strcmp -fno-builtin-strncmp \
// RUN:  -fno-builtin-strcpy -fno-builtin-stpcpy -fno-builtin-strncpy -fno-builtin-strlen \
// RUN:  -fno-builtin-strpbrk -fno-builtin-strspn -fno-builtin-strtod -fno-builtin-strtof \
// RUN:  -fno-builtin-strtold -fno-builtin-strtol -fno-builtin-strtoll -fno-builtin-strtoul \
// RUN:  -fno-builtin-strtoull -fno-builtin-fread -fno-builtin-fwrite -fno-builtin-fopen \
// RUN:  -o - %s | FileCheck --check-prefixes=INDIVIDUAL,CHECK %s
// RUN: %clang_cc1 -S -O3 -fno-builtin -o - %s | FileCheck --check-prefix=ASM %s
// RUN: %clang_cc1 -S -O3 -fno-builtin-ceil -o - %s | FileCheck --check-prefix=ASM-INDIV %s

typedef __SIZE_TYPE__ size_t;
typedef struct FILE FILE;

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
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb,
              FILE *stream);
FILE *fopen(const char *path, const char *mode);

double t1(double x) { return ceil(x); }
// CHECK-LABEL: t1
// CHECK: call{{.*}}@ceil{{.*}} [[ATTR:#[0-9]+]]

// ASM: t1
// ASM: ceil

// ASM-INDIV: t1
// ASM-INDIV: ceil

double t2(double x, double y) { return copysign(x,y); }
// CHECK-LABEL: t2
// CHECK: call{{.*}}@copysign{{.*}} #2

double t3(double x) { return cos(x); }
// CHECK-LABEL: t3
// CHECK: call{{.*}}@cos{{.*}} #2

double t4(double x) { return fabs(x); }
// CHECK-LABEL: t4
// CHECK: call{{.*}}@fabs{{.*}} #2

double t5(double x) { return floor(x); }
// CHECK-LABEL: t5
// CHECK: call{{.*}}@floor{{.*}} #2

char *t6(char *x) { return strcat(x, ""); }
// CHECK-LABEL: t6
// CHECK: call{{.*}}@strcat{{.*}} #2

char *t7(char *x) { return strncat(x, "", 1); }
// CHECK-LABEL: t7
// CHECK: call{{.*}}@strncat{{.*}} #2

char *t8(void) { return strchr("hello, world", 'w'); }
// CHECK-LABEL: t8
// CHECK: call{{.*}}@strchr{{.*}} #2

char *t9(void) { return strrchr("hello, world", 'w'); }
// CHECK-LABEL: t9
// CHECK: call{{.*}}@strrchr{{.*}} #2

int t10(void) { return strcmp("foo", "bar"); }
// CHECK-LABEL: t10
// CHECK: call{{.*}}@strcmp{{.*}} #2

int t11(void) { return strncmp("foo", "bar", 3); }
// CHECK-LABEL: t11
// CHECK: call{{.*}}@strncmp{{.*}} #2

char *t12(char *x) { return strcpy(x, "foo"); }
// CHECK-LABEL: t12
// CHECK: call{{.*}}@strcpy{{.*}} #2

char *t13(char *x) { return stpcpy(x, "foo"); }
// CHECK-LABEL: t13
// CHECK: call{{.*}}@stpcpy{{.*}} #2

char *t14(char *x) { return strncpy(x, "foo", 3); }
// CHECK-LABEL: t14
// CHECK: call{{.*}}@strncpy{{.*}} #2

size_t t15(void) { return strlen("foo"); }
// CHECK-LABEL: t15
// CHECK: call{{.*}}@strlen{{.*}} #2

char *t16(char *x) { return strpbrk(x, ""); }
// CHECK-LABEL: t16
// CHECK: call{{.*}}@strpbrk{{.*}} #2

size_t t17(char *x) { return strspn(x, ""); }
// CHECK-LABEL: t17
// CHECK: call{{.*}}@strspn{{.*}} #2

double t18(char **x) { return strtod("123.4", x); }
// CHECK-LABEL: t18
// CHECK: call{{.*}}@strtod{{.*}} #2

float t19(char **x) { return strtof("123.4", x); }
// CHECK-LABEL: t19
// CHECK: call{{.*}}@strtof{{.*}} #2

long double t20(char **x) { return strtold("123.4", x); }
// CHECK-LABEL: t20
// CHECK: call{{.*}}@strtold{{.*}} #2

long int t21(char **x) { return strtol("1234", x, 10); }
// CHECK-LABEL: t21
// CHECK: call{{.*}}@strtol{{.*}} #2

long int t22(char **x) { return strtoll("1234", x, 10); }
// CHECK-LABEL: t22
// CHECK: call{{.*}}@strtoll{{.*}} #2

long int t23(char **x) { return strtoul("1234", x, 10); }
// CHECK-LABEL: t23
// CHECK: call{{.*}}@strtoul{{.*}} #2

long int t24(char **x) { return strtoull("1234", x, 10); }
// CHECK-LABEL: t24
// CHECK: call{{.*}}@strtoull{{.*}} #2

void t25(FILE *fp, int *buf) {
  size_t x = fwrite(buf, sizeof(int), 10, fp);
  size_t y = fread(buf, sizeof(int), 10, fp);
}
// CHECK-LABEL: t25
// CHECK: call{{.*}}@fwrite{{.*}} #2
// CHECK: call{{.*}}@fread{{.*}} #2

FILE *t26(const char *path, const char *mode) {
  return fopen(path, mode);
}
// CHECK-LABEL: t26
// CHECK: call{{.*}}@fopen{{.*}} #2

// GLOBAL: #2 = { nobuiltin "no-builtins" }
// INDIVIDUAL: #2 = { nobuiltin "no-builtin-ceil" "no-builtin-copysign" "no-builtin-cos" "no-builtin-fabs" "no-builtin-floor" "no-builtin-fopen" "no-builtin-fread" "no-builtin-fwrite" "no-builtin-stpcpy" "no-builtin-strcat" "no-builtin-strchr" "no-builtin-strcmp" "no-builtin-strcpy" "no-builtin-strlen" "no-builtin-strncat" "no-builtin-strncmp" "no-builtin-strncpy" "no-builtin-strpbrk" "no-builtin-strrchr" "no-builtin-strspn" "no-builtin-strtod" "no-builtin-strtof" "no-builtin-strtol" "no-builtin-strtold" "no-builtin-strtoll" "no-builtin-strtoul" "no-builtin-strtoull" }
