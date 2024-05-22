// RUN: %clang_cc1 -fsanitize-trap=returns-nonnull-attribute -fsanitize=returns-nonnull-attribute -emit-llvm %s -o - -triple x86_64-apple-darwin10 -fblocks | FileCheck %s

// Awkward interactions of sanitizers with blocks.

const char *TheString = "Hello, world!";
const char *(^getString)(void) = ^{
  return TheString;
};

// CHECK-LABEL: define internal ptr @getString_block_invoke

// TODO: Actually support returns_nonnull on blocks.
