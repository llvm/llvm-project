// RUN: %clang_cc1 -no-opaque-pointers -fsanitize-trap=returns-nonnull-attribute -fsanitize=returns-nonnull-attribute -emit-llvm %s -o - -triple x86_64-apple-darwin10 -fblocks | FileCheck %s

// Awkward interactions of sanitizers with blocks.

// rdar://22071955
const char *TheString = "Hello, world!";
const char *(^getString)(void) = ^{
  return TheString;
};

// CHECK-LABEL: define internal i8* @getString_block_invoke

// TODO: Actually support returns_nonnull on blocks.
