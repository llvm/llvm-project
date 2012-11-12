// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s

// Tests that -ffreestanding disables all special treatment of main().

void* allocate(long size);

void* main(void* context, long size) {
  if (context) return allocate(size);
} // expected-warning {{control may reach end of non-void function}}
