// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR4390
struct sysfs_dirent {
 union { struct sysfs_elem_dir { int x; } s_dir; };
 unsigned short s_mode;
};
struct sysfs_dirent sysfs_root = { {}, 16877 };

// CHECK: @sysfs_root = global %struct.sysfs_dirent { %union.anon zeroinitializer, i16 16877 }

struct Foo {
 union { struct empty {} x; };
 unsigned short s_mode;
};
struct Foo foo = { {}, 16877 };

// CHECK: @foo = global %struct.Foo { i16 16877 }
