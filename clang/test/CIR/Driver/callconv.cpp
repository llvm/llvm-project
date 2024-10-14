// RUN: %clang %s -fno-clangir-call-conv-lowering -### -c %s 2>&1 | FileCheck --check-prefix=DISABLE %s
// DISABLE: "-fno-clangir-call-conv-lowering"
// RUN: %clang %s -fclangir-call-conv-lowering -### -c %s 2>&1 | FileCheck --check-prefix=ENABLE %s
// ENABLE-NOT: "-fclangir-call-conv-lowering"
