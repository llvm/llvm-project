// RUN: %clang %s -fclangir-move-opt -### -c %s 2>&1 | FileCheck --check-prefix=ENABLE %s
// ENABLE: "-fclangir-move-opt"
