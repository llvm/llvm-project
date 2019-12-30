// RUN: %clang -target s390x -c -### %s -mpacked-stack -mbackchain 2>&1 | FileCheck %s

// CHECK: error: unsupported option '-mpacked-stack -mbackchain'
