// RUN: %clang_cc1 -std=c++11 -emit-llvm -o %t-c++11.ll %s -triple x86_64-apple-darwin10
// RUN: FileCheck %s < %t-c++11.ll
// RUN: %clang_cc1 -std=c++17 -emit-llvm -o %t-c++17.ll %s -triple x86_64-apple-darwin10
// RUN: FileCheck %s < %t-c++17.ll
// RUN: %clang_cc1  -std=c++98 -emit-llvm -o %t.ll %s -triple x86_64-apple-darwin10
// RUN: %clang_cc1  -std=c++03 -emit-llvm -o %t-c++03.ll %s -triple x86_64-apple-darwin10
// RUN: diff %t-c++11.ll  %t-c++17.ll
// RUN: diff %t.ll  %t-c++03.ll

struct sAFSearchPos {
    unsigned char *pos;
    unsigned char count;
};

static volatile struct sAFSearchPos testPositions;
// CHECK: @_ZL13testPositions = internal global %struct.sAFSearchPos zeroinitializer

static volatile struct sAFSearchPos arrayPositions[100][10][5];
// CHECK: @_ZL14arrayPositions = internal global [100 x [10 x [5 x %struct.sAFSearchPos]]] zeroinitializer

int main() {
  return testPositions.count + arrayPositions[10][4][3].count; 
}
