

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -ast-dump %s | FileCheck %s
#include <ptrcheck.h>

const int * auto_global_var1;
// CHECK: auto_global_var1 'const int *__single'

const int * _Nullable auto_global_var2;
// CHECK: auto_global_var2 'const int *__single _Nullable':'const int *__single'

const int * __ptrauth(2, 0, 0) auto_global_var3;
// CHECK: auto_global_var3 'const int *__single__ptrauth(2,0,0)'

const int * __single global_var1;
// CHECK: global_var1 'const int *__single'

const int * _Nullable __single global_var2;
// CHECK: global_var2 'const int *__single _Nullable':'const int *__single'

const int * __ptrauth(2, 0, 0) __single global_var3;
// CHECK: global_var3 'const int *__single__ptrauth(2,0,0)'


const int * _Nullable __ptrauth(2, 0, 0) __single c1_global_var1;
// CHECK: c1_global_var1 'const int *__single _Nullable __ptrauth(2,0,0)':'const int *__single__ptrauth(2,0,0)'

const int * _Nullable __single __ptrauth(2, 0, 0) c1_global_var2;
// CHECK: c1_global_var2 'const int *__single _Nullable __ptrauth(2,0,0)':'const int *__single__ptrauth(2,0,0)'

const int * __ptrauth(2, 0, 0) _Nullable __single c1_global_var3;
// CHECK: c1_global_var3 'const int *__single__ptrauth(2,0,0) _Nullable':'const int *__single__ptrauth(2,0,0)'

const int * __ptrauth(2, 0, 0) __single _Nullable c1_global_var4;
// CHECK: c1_global_var4 'const int *__single__ptrauth(2,0,0) _Nullable':'const int *__single__ptrauth(2,0,0)'

const int * __single __ptrauth(2, 0, 0) _Nullable c1_global_var5;
// CHECK: c1_global_var5 'const int *__single__ptrauth(2,0,0) _Nullable':'const int *__single__ptrauth(2,0,0)'

const int * __single _Nullable __ptrauth(2, 0, 0) c1_global_var6;
// CHECK: c1_global_var6 'const int *__single _Nullable __ptrauth(2,0,0)':'const int *__single__ptrauth(2,0,0)'


int * const _Nullable __ptrauth(2, 0, 0) __single c2_global_var1;
// CHECK: c2_global_var1 'int *__singleconst  _Nullable __ptrauth(2,0,0)':'int *__singleconst __ptrauth(2,0,0)'

int * const _Nullable __single __ptrauth(2, 0, 0) c2_global_var2;
// CHECK: c2_global_var2 'int *__singleconst  _Nullable __ptrauth(2,0,0)':'int *__singleconst __ptrauth(2,0,0)'

int * const __ptrauth(2, 0, 0) _Nullable __single c2_global_var3;
// CHECK: c2_global_var3 'int *__singleconst __ptrauth(2,0,0) _Nullable':'int *__singleconst __ptrauth(2,0,0)'

int * const __ptrauth(2, 0, 0) __single _Nullable c2_global_var4;
// CHECK: c2_global_var4 'int *__singleconst __ptrauth(2,0,0) _Nullable':'int *__singleconst __ptrauth(2,0,0)'

int * const __single __ptrauth(2, 0, 0) _Nullable c2_global_var5;
// CHECK: c2_global_var5 'int *__singleconst __ptrauth(2,0,0) _Nullable':'int *__singleconst __ptrauth(2,0,0)'

int * const __single _Nullable __ptrauth(2, 0, 0) c2_global_var6;
// CHECK: c2_global_var6 'int *__singleconst _Nullable __ptrauth(2,0,0)':'int *__singleconst __ptrauth(2,0,0)'


const int * const _Nullable __ptrauth(2, 0, 0) __single c12_global_var1;
// CHECK: c12_global_var1 'const int *__singleconst _Nullable __ptrauth(2,0,0)':'const int *__singleconst __ptrauth(2,0,0)'
const int * const _Nullable __single __ptrauth(2, 0, 0) c12_global_var2;
// CHECK: c12_global_var2 'const int *__singleconst  _Nullable __ptrauth(2,0,0)':'const int *__singleconst __ptrauth(2,0,0)'

const int * const __ptrauth(2, 0, 0) _Nullable __single c12_global_var3;
// CHECK: c12_global_var3 'const int *__singleconst __ptrauth(2,0,0) _Nullable':'const int *__singleconst __ptrauth(2,0,0)'

const int * const __ptrauth(2, 0, 0) __single _Nullable c12_global_var4;
// CHECK: c12_global_var4 'const int *__singleconst __ptrauth(2,0,0) _Nullable':'const int *__singleconst __ptrauth(2,0,0)'

const int * const __single __ptrauth(2, 0, 0) _Nullable c12_global_var5;
// CHECK: c12_global_var5 'const int *__singleconst __ptrauth(2,0,0) _Nullable':'const int *__singleconst __ptrauth(2,0,0)'

const int * const __single _Nullable __ptrauth(2, 0, 0) c12_global_var6;
// CHECK: c12_global_var6 'const int *__singleconst _Nullable __ptrauth(2,0,0)':'const int *__singleconst __ptrauth(2,0,0)'
