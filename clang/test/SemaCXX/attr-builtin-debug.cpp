// RUN: %clang_cc1 -fsyntax-only -verify %s -ffreestanding -Wno-c++23-extensions

[[clang::builtin("memcpy")]] void func(); // expected-error {{function signature does not match the signature of the builtin}} \
                                             expected-note {{expected signature is 'void *(void *, const void *, unsigned long)'}}
