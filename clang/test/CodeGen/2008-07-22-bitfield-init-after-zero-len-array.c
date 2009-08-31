// RUN: clang-cc -triple=i686-apple-darwin9 --emit-llvm -o - %s | FileCheck %s

struct et7 {
  float lv7[0];
  char mv7:6;
} yv7 = {
  {}, 
  52, 
};

// CHECK: @yv7 = global 
// CHECK: i8 52,