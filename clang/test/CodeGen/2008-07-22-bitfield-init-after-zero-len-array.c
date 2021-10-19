// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

struct et7 {
  float lv7[0];
  char mv7:6;
} yv7 = {
  {}, 
  52, 
};

// CHECK: @yv7 ={{.*}} global %struct.et7 { [0 x float] zeroinitializer, i8 52 }
