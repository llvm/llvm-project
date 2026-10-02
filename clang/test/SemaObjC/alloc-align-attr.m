// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface AllocAlignMethod
- (void *)allocate:(unsigned long)alignment
    __attribute__((alloc_align(1)));
- (void *)allocateInvalid:(float)alignment
    __attribute__((alloc_align(1))); // expected-error {{'alloc_align' attribute argument may only refer to a function parameter of integer type}}
@end
