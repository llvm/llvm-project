// RUN: %clang_cc1 -Wconversion -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wbitfield-conversion -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple armebv7-unknown-linux -Wbitfield-conversion \
// RUN:     -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-unknown-linux  -Wbitfield-conversion \
// RUN:     -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm-unknown-linux -Wbitfield-conversion \
// RUN:     -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux -Wbitfield-conversion \
// RUN:     -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mipsel-unknown-linux -Wbitfield-conversion \
// RUN:     -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux -Wbitfield-conversion \
// RUN:     -fsyntax-only -verify %s

typedef struct _xx {
     int bf:9; // expected-note 3{{declared here}}
 } xx, *pxx; 

 xx vxx;

 void foo1(int x) {     
     vxx.bf = x; // expected-warning{{conversion from 'int' (32 bits) to bit-field 'bf' (9 bits) may change value}}
 } 
 void foo2(short x) {     
     vxx.bf = x; // expected-warning{{conversion from 'short' (16 bits) to bit-field 'bf' (9 bits) may change value}}
 } 
 void foo3(char x) {     
     vxx.bf = x; // no warning expected
 } 
 void foo4(short x) {     
     vxx.bf = 0xff & x; // no warning expected 
 } 
 void foo5(short x) {     
     vxx.bf = 0x1ff & x; // no warning expected 
 } 
 void foo6(short x) {     
     vxx.bf = 0x3ff & x; // expected-warning{{conversion from 'int' (10 bits) to bit-field 'bf' (9 bits) may change value}}
 } 
 int fee(void) {
     return 0;
 }
