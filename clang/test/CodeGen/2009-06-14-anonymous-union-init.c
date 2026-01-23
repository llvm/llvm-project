// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefixes=CHECK,EMPTY
// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-windows-msvc -o - | FileCheck %s --check-prefixes=CHECK,EMPTY-MSVC
// PR4390
struct sysfs_dirent {
 union { struct sysfs_elem_dir { int x; } s_dir; };
 unsigned short s_mode;
};
struct sysfs_dirent sysfs_root = { {}, 16877 };

// CHECK: @sysfs_root = {{.*}}global { %union.anon, i16, [2 x i8] } { %union.anon zeroinitializer, i16 16877, [2 x i8] zeroinitializer }

struct Foo {
 union { struct empty {} x; };
 unsigned short s_mode;
};
struct Foo foo = { {}, 16877 };

// EMPTY:      @foo = {{.*}}global %struct.Foo { i16 16877 }
// EMPTY-MSVC: @foo = {{.*}}global %struct.Foo { [4 x i8] zeroinitializer, i16 16877 }
