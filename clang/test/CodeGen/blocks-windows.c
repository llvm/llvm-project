// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DECL -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DECL
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DEFN -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DEFN
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -static-libclosure -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_DLLIMPORT -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT

// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DECL -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DECL
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DEFN -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DEFN
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -Os -static-libclosure -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple thumbv7-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_DLLIMPORT -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT

// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DECL -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DECL
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DEFN -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DEFN
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -static-libclosure -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_DLLIMPORT -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT

// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DECL -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DECL
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DEFN -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DEFN
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -Os -static-libclosure -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple i686-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_DLLIMPORT -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT

// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DECL -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DECL
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DEFN -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DEFN
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -static-libclosure -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_DLLIMPORT -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT

// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DECL -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DECL
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_IN_BLOCKS_DEFN -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-IN-BLOCKS-DEFN
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -Os -static-libclosure -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple x86_64-windows -fblocks -fdeclspec -DBLOCKS_NOT_IN_BLOCKS_DLLIMPORT -Os -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT

void *_Block_copy(void *);

#if defined(BLOCKS_IN_BLOCKS_DECL)
extern __declspec(dllexport) long _NSConcreteStackBlock[];
#endif

#if defined(BLOCKS_IN_BLOCKS_DEFN)
__declspec(dllexport) long _NSConcreteStackBlock[5];
#endif

#if defined(BLOCKS_NOT_IN_BLOCKS_EXTERN)
extern long _NSConcreteStackBlock[];
#endif

#if defined(BLOCKS_NOT_IN_BLOCKS_EXTERN_DLLIMPORT)
extern __declspec(dllimport) long _NSConcreteStackBlock[];
#endif

#if defined(BLOCKS_NOT_IN_BLOCKS_DLLIMPORT)
__declspec(dllimport) long _NSConcreteStackBlock[];
#endif

int (*g(void))(void) {
  __block int i;
  return _Block_copy(^{ ++i; return i; });
}

// CHECK-BLOCKS-IN-BLOCKS-DECL: @_NSConcreteStackBlock = external dso_local dllexport global ptr
// CHECK-BLOCKS-IN-BLOCKS-DEFN: @_NSConcreteStackBlock = dso_local dllexport global [5 x i32]
// CHECK-BLOCKS-NOT-IN-BLOCKS: @_NSConcreteStackBlock = external dllimport global ptr
// CHECK-BLOCKS-NOT-IN-STATIC-BLOCKS: @_NSConcreteStackBlock = external dso_local global ptr
// CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN: @_NSConcreteStackBlock = external dllimport global ptr
// CHECK-BLOCKS-NOT-IN-BLOCKS-EXTERN-DLLIMPORT: @_NSConcreteStackBlock = external dllimport global ptr
// CHECK-BLOCKS-NOT-IN-BLOCKS-DLLIMPORT: @_NSConcreteStackBlock = external dllimport global ptr

