// RUN: %clang_cc1 -emit-llvm < %s | grep "zeroinitializer, i16 16877"
// PR4390
struct sysfs_dirent {
 union { struct sysfs_elem_dir {} s_dir; };
 unsigned short s_mode;
};
struct sysfs_dirent sysfs_root = { {}, 16877 };
