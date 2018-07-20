// RUN: %clang -emit-llvm -S -g -gcodeview -x c %S/Inputs/debug-info-file-checksum.c -o - | FileCheck %s
// RUN: %clang -emit-llvm -S -gdwarf-5 -x c %S/Inputs/debug-info-file-checksum.c -o - | FileCheck %s

// Check that "checksum" is created correctly for the compiled file.

// CHECK: !DIFile(filename:{{.*}}, directory:{{.*}}, checksumkind: CSK_MD5, checksum: "a3b7d27af071accdeccaa933fc603608")
