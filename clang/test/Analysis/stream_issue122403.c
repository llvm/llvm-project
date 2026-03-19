// RUN: %clang_analyze_cc1 -triple=x86_64-pc-linux-gnu -analyzer-checker=core,unix.Stream,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false,unix.Stream:Pedantic=true -verify %s
// RUN: %clang_analyze_cc1 -triple=armv8-none-linux-eabi -analyzer-checker=core,unix.Stream,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false,unix.Stream:Pedantic=true -verify %s
// RUN: %clang_analyze_cc1 -triple=aarch64-linux-gnu -analyzer-checker=core,unix.Stream,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false,unix.Stream:Pedantic=true -verify %s
// RUN: %clang_analyze_cc1 -triple=hexagon -analyzer-checker=core,unix.Stream,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false,unix.Stream:Pedantic=true -verify %s

#include "Inputs/system-header-simulator.h"

char *get_str(char *Input);

void check_f_leak() {
  FILE *fp = fopen("test", "rb");
  if (NULL == fp) {
    return;
  }
  char str[64];
  if (get_str(str) != str) {
    fclose(fp);
  }
}// expected-warning {{Opened stream never closed. Potential resource leak}}

void check_f_leak_2() {
  FILE *fp = fopen("test", "rb");
  if (NULL == fp) {
    return;
  }
  char str[64];
  if (get_str(str) != NULL) {
    fclose(fp);
  }
}// expected-warning {{Opened stream never closed. Potential resource leak}}


char *get_str_other(char *Input) {return Input;}

void check_f_leak_3() {
  FILE *fp = fopen("test", "rb");
  if (NULL == fp) {
    return;
  }
  char str[64];
  if (get_str_other(str) != str) {
    fclose(fp);
  }
}// expected-warning {{Opened stream never closed. Potential resource leak}}