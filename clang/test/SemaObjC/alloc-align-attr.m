// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

@interface AllocAlignMethod
- (void *)allocate:(unsigned long)alignment
    __attribute__((alloc_align(1)));
@end
