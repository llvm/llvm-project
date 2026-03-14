// RUN: %clang_cc1 -Werror=return-type -triple i686-pc-openbsd -fsyntax-only -verify -ffreestanding %s

// Tests that -ffreestanding disables all special treatment of main().

void* allocate(long size);

void* main(void* context, long size) {
  if (context) return allocate(size);
} // expected-error {{non-void function does not return a value in all control paths}}
