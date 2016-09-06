// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=i686-apple-darwin9
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=arm-linux-gnueabihf
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=aarch64-linux-gnu
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple=x86_64-pc-linux-gnu
// expected-no-diagnostics

#define CHECK_SIZE(name, size) \
  extern int name##_1[sizeof(name) == size ? 1 : -1];


struct  __attribute__((packed)) {
  int a;
  int b : 4;
  int c : 32;
} s0;
CHECK_SIZE(s0,9)

#pragma pack (1)
struct {
  int a;
  int b : 4;
  int c : 32;
} s1;
CHECK_SIZE(s1,9)

#pragma pack (2)
struct {
  int a;
  int b : 4;
  int c : 32;
} s2;
CHECK_SIZE(s2,10)

#pragma pack (2)
struct __attribute__((packed)) {
  int a;
  int b : 4;
  int c : 32;
} s3;
CHECK_SIZE(s3,10)

#pragma pack (4)
struct  __attribute__((packed)) {
  int a;
  int b : 4;
  int c : 32;
} s4;
CHECK_SIZE(s4,12)

#pragma pack (16)
struct {
  int a;
  int __attribute__((packed)) b : 4;
  int __attribute__((packed)) c : 32;
} s41;
CHECK_SIZE(s41,12)

#pragma pack (16)
struct {
  int a;
  int b : 4;
  int c : 32;
} s5;
CHECK_SIZE(s5,12)

#pragma pack (1)
struct  __attribute__((aligned(4))) {
  int a;
  int b : 4;
  int c : 32;
} s6;
CHECK_SIZE(s6,12)

#pragma pack (2)
struct {
  char a;
  int b : 4;
  int c : 32;
  char s;
} s7;
CHECK_SIZE(s7,8)

#pragma pack (1)
struct {
  char a;
  int b : 4;
  int c : 28;
  char s;
} s8;
CHECK_SIZE(s8,6)

#pragma pack (8)
struct {
  char a;
  int b : 4;
  int c : 28;
  char s;
} s9;
CHECK_SIZE(s9,8)

#pragma pack (8)
struct {
  char a;
  char s;
} s10;
CHECK_SIZE(s10,2)

#pragma pack(4)
struct {
  char a;
  int b : 4;
  int c : 28;
  char s1;
  char s2;
  char s3;
} s11;
CHECK_SIZE(s11,8)

#pragma pack(4)
struct {
  short s1;
  int a1 : 17;
  int a2 : 17;
  int a3 : 30;
  short s2;
} s12;
CHECK_SIZE(s12,12)

#pragma pack(4)
struct {
  char c1;
  int i1 : 17;
  int i2 : 17;
  int i3 : 30;
  char c2;
} s13;
CHECK_SIZE(s13,12)

#pragma pack(2)
struct {
  char a;
  int s;
} s14;
CHECK_SIZE(s14,6)

#pragma pack(4)
struct {
  char a;
  short s;
} s15;
CHECK_SIZE(s15,4)

#pragma pack(2)
struct {
  char a;
  int b : 4;
  int c : 28;
  char s1;
  char s2;
  char s3;
} s16;
CHECK_SIZE(s16,8)

#pragma pack (16)
struct {
  int __attribute__((packed)) a;
  int __attribute__((packed)) b : 4;
  int __attribute__((packed)) c : 32;
} s17;
CHECK_SIZE(s17,12)

#pragma pack (16)
struct {
  int __attribute__((aligned(8))) a;
  int __attribute__((aligned(8))) b : 4;
  int __attribute__((aligned(8))) c : 32;
} s18;
CHECK_SIZE(s18,24)

#pragma pack (16)
struct {
  int __attribute__((aligned(1))) a;
  int __attribute__((aligned(1))) b : 4;
  int __attribute__((aligned(1))) c : 32;
} s19;
CHECK_SIZE(s19,12)

#pragma pack (1)
struct  __attribute__((aligned(8))) {
  int a;
  int b : 4;
  int c : 32;
} s20;
CHECK_SIZE(s20,16)

#pragma pack (2)
struct {
  int __attribute__((aligned(8))) a;
  int __attribute__((aligned(8))) b : 4;
  int __attribute__((aligned(8))) c : 32;
} s21;
CHECK_SIZE(s21,10)
