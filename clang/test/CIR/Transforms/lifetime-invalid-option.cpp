// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-lifetime-check="yolo=invalid,null" -emit-cir %s -o - 2>&1 | FileCheck %s

// CHECK: clangir pass option 'yolo=invalid,null' not recognized