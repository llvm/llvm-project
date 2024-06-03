// RUN: %clang %s -fclangir-analysis-only -### -c %s 2>&1 | FileCheck %s
// CHECK: "-fclangir-analysis-only"
