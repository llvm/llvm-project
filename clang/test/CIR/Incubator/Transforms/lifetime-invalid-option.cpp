// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="yolo=invalid,null" -emit-cir %s -o - 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-idiom-recognizer="idiom=invalid" -emit-cir %s -o - 2>&1 | FileCheck %s --check-prefix=IDIOM
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-idiom-recognizer -fclangir-lib-opt="libopt=invalid" -emit-cir %s -o - 2>&1 | FileCheck %s --check-prefix=LIBOPT

// CHECK: clangir pass option 'yolo=invalid,null' not recognized
// IDIOM: clangir pass option 'idiom=invalid' not recognized
// LIBOPT: clangir pass option 'libopt=invalid' not recognized
