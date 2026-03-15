// RUN: %clang_cc1 -std=c23 %s -E -verify --embed-dir=%S --embed-dir=%S/Inputs
// expected-no-diagnostics

#embed <jk.txt>
