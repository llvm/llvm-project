// RUN: %clang %s -fclangir-idiom-recognizer -### -c %s 2>&1 | FileCheck --check-prefix=ENABLE %s
// ENABLE: "-fclangir-idiom-recognizer"
