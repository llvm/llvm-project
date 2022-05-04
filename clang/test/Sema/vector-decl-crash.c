// RUN: %clang_cc1 %s -fsyntax-only -verify -triple x86_64-unknown-unknown

// GH50171
// This would previously crash when __bf16 was not a supported type.
__bf16 v64bf __attribute__((vector_size(128))); // expected-error {{__bf16 is not supported on this target}} \
                                                   expected-error {{vector size not an integral multiple of component size}}

