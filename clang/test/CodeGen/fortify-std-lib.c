// RUN: %clang_cc1       -triple x86_64-apple-macosx10.14.0 %s -emit-llvm -O0 -disable-llvm-passes -o - -Wno-format-security | FileCheck %s
// RUN: %clang_cc1 -xc++ -triple x86_64-apple-macosx10.14.0 %s -emit-llvm -O0 -disable-llvm-passes -o - -Wno-format-security | FileCheck %s

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

#define FSL(x,y) __attribute__((fortify_stdlib(x,y)))
typedef unsigned long size_t;

FSL(0, 0) EXTERN
void *memcpy(void *dst, const void *src, size_t sz);

EXTERN
void call_memcpy(void *dst, const void *src, size_t sz) {
  memcpy(dst, src, sz);
  // CHECK-LABEL: define void @call_memcpy
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__memcpy_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
void *memmove(void *dst, const void *src, size_t sz);

EXTERN
void call_memmove(void *dst, const void *src, size_t sz) {
  memmove(dst, src, sz);
  // CHECK-LABEL: define void @call_memmove
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__memmove_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
void *memset(void *dst, int c, size_t sz);

EXTERN
void call_memset(void *dst, int c, size_t sz) {
  memset(dst, c, sz);
  // CHECK-LABEL: define void @call_memset
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__memset_chk(i8* {{.*}}, i32 {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
char *stpcpy(char* dst, const char *src);

EXTERN
void call_stpcpy(char *dst, const char *src) {
  stpcpy(dst, src);
  // CHECK-LABEL: define void @call_stpcpy
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__stpcpy_chk(i8* {{.*}}, i8* {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
char *strcat(char* dst, const char *src);

EXTERN
void call_strcat(char *dst, const char *src) {
  strcat(dst, src);
  // CHECK-LABEL: define void @call_strcat
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__strcat_chk(i8* {{.*}}, i8* {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
char *strcpy(char* dst, const char *src);

EXTERN
void call_strcpy(char *dst, const char *src) {
  strcpy(dst, src);
  // CHECK-LABEL: define void @call_strcpy
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__strcpy_chk(i8* {{.*}}, i8* {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
size_t strlcat(char* dst, const char *src, size_t len);

EXTERN
void call_strlcat(char *dst, const char *src, size_t len) {
  strlcat(dst, src, len);
  // CHECK-LABEL: define void @call_strlcat
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i64 @__strlcat_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
size_t strlcpy(char* dst, const char *src, size_t len);

EXTERN
void call_strlcpy(char *dst, const char *src, size_t len) {
  strlcpy(dst, src, len);
  // CHECK-LABEL: define void @call_strlcpy
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i64 @__strlcpy_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
char *strncat(char* dst, const char *src, size_t len);

EXTERN
void call_strncat(char *dst, const char *src, size_t len) {
  strncat(dst, src, len);
  // CHECK-LABEL: define void @call_strncat
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__strncat_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
char *strncpy(char* dst, const char *src, size_t len);

EXTERN
void call_strncpy(char *dst, const char *src, size_t len) {
  strncpy(dst, src, len);
  // CHECK-LABEL: define void @call_strncpy
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__strncpy_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
char *stpncpy(char* dst, const char *src, size_t len);

EXTERN
void call_stpncpy(char *dst, const char *src, size_t len) {
  stpncpy(dst, src, len);
  // CHECK-LABEL: define void @call_stpncpy
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i8* @__stpncpy_chk(i8* {{.*}}, i8* {{.*}}, i64 {{.*}}, i64 [[REG]])
}

FSL(0, 0) EXTERN
int snprintf(char *buf, size_t n, const char *fmt, ...);

EXTERN
void call_snprintf(char *buf, size_t n, const char *fmt) {
  snprintf(buf, n, fmt);
  // CHECK-LABEL: define void @call_snprintf
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i32 (i8*, i64, i32, i64, i8*, ...) @__snprintf_chk(i8* {{.*}}, i64 {{.*}}, i32 0, i64 [[REG]]
}

FSL(0, 0) EXTERN
int vsnprintf(char *buf, size_t n, const char *fmt, __builtin_va_list lst);

EXTERN
void call_vsnprintf(char *buf, size_t n, const char *fmt, __builtin_va_list lst) {
  vsnprintf(buf, n, fmt, lst);
  // CHECK-LABEL: define void @call_vsnprintf
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i32 @__vsnprintf_chk(i8* {{.*}}, i64 {{.*}}, i32 0, i64 [[REG]]
}

FSL(0,0) EXTERN
int sprintf(char *buf, const char *fmt, ...);

void call_sprintf(char *buf, const char* fmt) {
  sprintf(buf, fmt);
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* {{.*}}, i32 0, i64 [[REG]]
  sprintf(buf, fmt, 1, 2, 3);
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i32 (i8*, i32, i64, i8*, ...) @__sprintf_chk(i8* {{.*}}, i32 0, i64 [[REG]], i8* {{.*}}, i32 1, i32 2, i32 3)
}

FSL(0, 0) EXTERN
int vsprintf(char *buf, const char *fmt, __builtin_va_list lst);

EXTERN
void call_vsprintf(char *buf, const char *fmt, __builtin_va_list lst) {
  vsprintf(buf, fmt, lst);
  // CHECK-LABEL: define void @call_vsprintf
  // CHECK: [[REG:%[0-9]+]] = call i64 @llvm.objectsize.i64.p0i8(i8*{{.*}}, i1 false, i1 true, i1 false)
  // CHECK: call i32 @__vsprintf_chk(i8* {{.*}}, i32 0, i64 [[REG]]
}

typedef struct {} FILE;

FSL(0, 0) EXTERN
int fprintf(FILE *file, const char *fmt, ...);

EXTERN
void call_fprintf(FILE *file, const char *fmt) {
  fprintf(file, fmt);
  // CHECK-LABEL: define void @call_fprintf
  // CHECK: call i32 ({{.*}}*, i32, i8*, ...) @__fprintf_chk({{.*}}, i32 0, i8* {{.*}})
}

FSL(0, 0) EXTERN
int vfprintf(FILE *file, const char *fmt, __builtin_va_list lst);

EXTERN
void call_vfprintf(FILE *file, const char *fmt, __builtin_va_list lst) {
  vfprintf(file, fmt, lst);
  // CHECK-LABEL: define void @call_vfprintf
  // CHECK: call i32 @__vfprintf_chk({{.*}}, i32 0, i8* {{.*}}, {{.*}})
}

FSL(0, 0) EXTERN
int printf(const char *fmt, ...);

EXTERN
void call_printf(const char *fmt) {
  printf(fmt);
  // CHECK-LABEL: define void @call_printf
  // CHECK: call i32 (i32, i8*, ...) @__printf_chk(i32 0, i8* {{.*}})
}

FSL(0, 0) EXTERN
int vprintf(const char *fmt, __builtin_va_list lst);

EXTERN
void call_vprintf(const char *fmt, __builtin_va_list lst) {
  vprintf(fmt, lst);
  // CHECK-LABEL: define void @call_vprintf
  // CHECK: call i32 @__vprintf_chk(i32 0, {{.*}})
}

