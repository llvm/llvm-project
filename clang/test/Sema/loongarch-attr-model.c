// RUN: %clang_cc1 -triple loongarch64 -verify -fsyntax-only %s

#if !__has_attribute(model)
#error "Should support model attribute"
#endif

int a __attribute((model("tiny"))); // expected-error {{code_model 'tiny' is not yet supported on this target}}
int b __attribute((model("small"))); // expected-error {{code_model 'small' is not yet supported on this target}}
int c __attribute((model("normal"))); // no-warning
int d __attribute((model("kernel"))); // expected-error {{code_model 'kernel' is not yet supported on this target}}
int e __attribute((model("medium"))); // no-warning
int f __attribute((model("large"))); // expected-error {{code_model 'large' is not yet supported on this target}}
int g __attribute((model("extreme"))); // no-warning
