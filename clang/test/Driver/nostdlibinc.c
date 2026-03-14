// RUN: %clang -target x86_64-unknown-unknown \
// RUN:   -nostdlibinc -ffreestanding -fsyntax-only %s

#if !__has_include("stddef.h")
#error "expected to be able to find compiler builtin headers!"
#endif

#if __has_include("stdlib.h")
#error "expected to *not* be able to find standard C headers"
#endif
