// RUN: %clang_cc1 -std=c++1z -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s -w | FileCheck %s

typedef int T;
void f() {
  // CHECK:      %[[A:.*]] = alloca i32, align 4
  // CHECK-NEXT: store i32 5, i32* %[[A]], align 4
  // CHECK-NEXT: %[[B:.*]] = load i32, i32* %[[A]], align 4
  // CHECK-NEXT: %[[C:.*]] = icmp slt i32 %[[B]], 8
  if (int a = 5; a < 8)
    ;
}

void f1() {
  // CHECK:      %[[A:.*]] = alloca i32, align 4
  // CHECK-NEXT: %[[B:.*]] = alloca i32, align 4
  // CHECK-NEXT: %[[C:.*]] = alloca i32, align 4
  // CHECK-NEXT: store i32 5, i32* %[[B]], align 4
  // CHECK-NEXT: store i32 7, i32* %[[C]], align 4
  if (int a, b = 5; int c = 7)
    ;
}

int f2() {
  // CHECK:      %[[A:.*]] = alloca i32, align 4
  // CHECK-NEXT: %[[B:.*]] = call i32 @_Z2f2v()
  // CHECK-NEXT: store i32 7, i32* %[[A]], align 4
  // CHECK-NEXT: %[[C:.*]] = load i32, i32* %[[A]], align 4
  // CHECK-NEXT: %[[D:.*]] = icmp ne i32 %[[C]], 0
  if (T{f2()}; int c = 7)
    ;
  return 2;
}

void g() {
  // CHECK:      %[[A:.*]] = alloca i32, align 4
  // CHECK-NEXT: store i32 5, i32* %[[A]], align 4
  // CHECK-NEXT: %[[B:.*]] = load i32, i32* %[[A]], align 4
  // CHECK-NEXT: switch i32 %[[B]], label %[[C:.*]] [
  switch (int a = 5; a) {
    case 0:
      break;
  }
}

void g1() {
  // CHECK:      %[[A:.*]] = alloca i32, align 4
  // CHECK-NEXT: %[[B:.*]] = alloca i32, align 4
  // CHECK-NEXT: %[[C:.*]] = alloca i32, align 4
  // CHECK-NEXT: store i32 5, i32* %[[B]], align 4
  // CHECK-NEXT: store i32 7, i32* %[[C]], align 4
  // CHECK-NEXT: %[[D:.*]] = load i32, i32* %[[C]], align 4
  // CHECK-NEXT: switch i32 %[[D]], label %[[E:.*]] [
  switch (int a, b = 5; int c = 7) {
    case 0:
      break;
  }
}

int g2() {
  // CHECK:      %[[A:.*]] = alloca i32, align 4
  // CHECK-NEXT: %[[B:.*]] = call i32 @_Z2f2v()
  // CHECK-NEXT: store i32 7, i32* %[[A]], align 4
  // CHECK-NEXT: %[[C:.*]] = load i32, i32* %[[A]], align 4
  // CHECK-NEXT: switch i32 %[[C]], label %[[E:.*]] [
  switch (T{f2()}; int c = 7) {
    case 0:
      break;
  }
  return 2;
}
