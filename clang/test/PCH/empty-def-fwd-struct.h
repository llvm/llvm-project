// RUN: %clang_cc1 -emit-pch -x c++-header %s -std=c++14 -o %t.pch
// RUN: %clang_cc1 -emit-llvm-only -x c++ /dev/null -std=c++14 -include-pch %t.pch -o %t.o
struct FVector;
struct FVector {};
struct FBox {
  FVector Min;
  FBox(int);
};
namespace {
FBox InvalidBoundingBox(0);
}

