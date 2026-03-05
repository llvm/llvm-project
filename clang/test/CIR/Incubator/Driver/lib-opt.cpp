// RUN: %clang %s -fclangir-lib-opt -### -c %s 2>&1 | FileCheck --check-prefix=ENABLE %s
// ENABLE: "-fclangir-lib-opt"
// ENABLE: "-fclangir-idiom-recognizer"
