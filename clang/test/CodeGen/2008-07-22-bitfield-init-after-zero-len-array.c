// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

struct et7 {
  float lv7[0];
  char mv7:6;
} yv7 = {
  {}, 
  52, 
};

// CHECK: @yv7 ={{.*}} global { [0 x float], i8, [3 x i8] } { [0 x float] zeroinitializer, i8 52, [3 x i8] zeroinitializer }
