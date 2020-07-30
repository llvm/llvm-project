// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

int checks_please() {
  return 1;
}

// UTC_ARGS: --disable

int no_checks_please() {
  // Manual CHECK line should be retained:
  // CHECK: manual check line
  return -1;
}

// UTC_ARGS: --enable


int checks_again() {
  return 2;
}
