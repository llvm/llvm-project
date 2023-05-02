// RUN: %clang_cc1 -debug-info-kind=standalone -fdebug-prefix-map=%p=./UNLIKELY_PATH/empty -S %s -emit-llvm -o - | FileCheck %s

struct alignas(64) an {
  struct {
    unsigned char x{0};
  } arr[64];
};

struct an *pan = new an;

// CHECK: !DISubprogram(name: "(unnamed struct at ./UNLIKELY_PATH/empty{{/|\\\\}}{{.*}}",
