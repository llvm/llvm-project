// RUN: %clang -### -working-directory /no/such/dir/ input 2>&1 | FileCheck %s

//CHECK: no such file or directory: '/no/such/dir/input'
