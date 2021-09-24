// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -fcir-warnings %s -fcir-output=%t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fsyntax-only -fcir-warnings %s -fcir-output=%t.cpp.cir
// RUN: FileCheck --input-file=%t.cpp.cir --check-prefix=CHECK-CPP %s
// XFAIL: *

int t0(int i) { return i; }
unsigned int t1(unsigned int i) { return i; }

char t2(char i) { return i; }
unsigned char t3(unsigned char i) { return i; }

short t4(short i) { return i; }
unsigned short t5(unsigned short i) { return i; }

float t6(float i) { return i; }
double t7(double i) { return i; }

void t8() {}

#ifdef __cplusplus
bool t9(bool b) { return b; }
#endif

// CHECK: func @t0(%arg0: i32) -> i32 {
// CHECK: func @t1(%arg0: i32) -> i32 {
// CHECK: func @t2(%arg0: i8) -> i8 {
// CHECK: func @t3(%arg0: i8) -> i8 {
// CHECK: func @t4(%arg0: i16) -> i16 {
// CHECK: func @t5(%arg0: i16) -> i16 {
// CHECK: func @t6(%arg0: f32) -> f32 {
// CHECK: func @t7(%arg0: f64) -> f64 {
// CHECK: func @t8() {

// CHECK-CPP: func @t0(%arg0: i32) -> i32 {
// CHECK-CPP: func @t1(%arg0: i32) -> i32 {
// CHECK-CPP: func @t2(%arg0: i8) -> i8 {
// CHECK-CPP: func @t3(%arg0: i8) -> i8 {
// CHECK-CPP: func @t4(%arg0: i16) -> i16 {
// CHECK-CPP: func @t5(%arg0: i16) -> i16 {
// CHECK-CPP: func @t6(%arg0: f32) -> f32 {
// CHECK-CPP: func @t7(%arg0: f64) -> f64 {
// CHECK-CPP: func @t8() {
// CHECK-CPP: func @t9(%arg0: i1) -> i1 {
