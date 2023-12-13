// RUN: %clang_cc1 -triple aarch64 -verify=expected,aarch64 -fsyntax-only %s
// RUN: %clang_cc1 -triple loongarch64 -verify=expected,loongarch64 -fsyntax-only %s
// RUN: %clang_cc1 -triple mips64 -verify=expected,mips64 -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64 -verify=expected,powerpc64 -fsyntax-only %s
// RUN: %clang_cc1 -triple riscv64 -verify=expected,riscv64 -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64 -verify=expected,x86_64 -fsyntax-only %s

#if !__has_attribute(model)
#error "Should support model attribute"
#endif

int a __attribute((model("tiny")));    // expected-error {{code model 'tiny' is not supported on this target}}
int b __attribute((model("small")));   // expected-error {{code model 'small' is not supported on this target}}
int c __attribute((model("normal")));  // aarch64-error {{code model 'normal' is not supported on this target}} \
                                       // mips64-error {{code model 'normal' is not supported on this target}} \
                                       // powerpc64-error {{code model 'normal' is not supported on this target}} \
                                       // riscv64-error {{code model 'normal' is not supported on this target}} \
                                       // x86_64-error {{code model 'normal' is not supported on this target}}
int d __attribute((model("kernel")));  // expected-error {{code model 'kernel' is not supported on this target}}
int e __attribute((model("medium")));  // aarch64-error {{code model 'medium' is not supported on this target}} \
                                       // mips64-error {{code model 'medium' is not supported on this target}} \
                                       // powerpc64-error {{code model 'medium' is not supported on this target}} \
                                       // riscv64-error {{code model 'medium' is not supported on this target}} \
                                       // x86_64-error {{code model 'medium' is not supported on this target}}
int f __attribute((model("large")));   // expected-error {{code model 'large' is not supported on this target}}
int g __attribute((model("extreme"))); // aarch64-error {{code model 'extreme' is not supported on this target}} \
                                       // mips64-error {{code model 'extreme' is not supported on this target}} \
                                       // powerpc64-error {{code model 'extreme' is not supported on this target}} \
                                       // riscv64-error {{code model 'extreme' is not supported on this target}} \
                                       // x86_64-error {{code model 'extreme' is not supported on this target}}
void __attribute((model("extreme"))) h() {} // expected-error {{'model' attribute only applies to non-TLS global variables}}
thread_local int i __attribute((model("extreme"))); // expected-error {{'model' attribute only applies to non-TLS global variables}}
