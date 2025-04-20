// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1                                         -verify=both,ref      %s

// both-no-diagnostics

int a;
static_assert(__builtin_object_size(&a, 0) == sizeof(int), "");
float f;
static_assert(__builtin_object_size(&f, 0) == sizeof(float), "");
int arr[2];
static_assert(__builtin_object_size(&arr, 0) == (sizeof(int)*2), "");

float arrf[2];
static_assert(__builtin_object_size(&arrf, 0) == (sizeof(float)*2), "");
