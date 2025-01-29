// RUN: %clang_cc1 -triple aarch64 -verify=expected,unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple loongarch64 -verify=expected,loongarch64 -fsyntax-only %s
// RUN: %clang_cc1 -triple mips64 -verify=expected,unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64 -verify=expected,unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple riscv64 -verify=expected,unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64 -verify=expected,x86_64 -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx64-unknown-cuda -fcuda-is-device -x cuda -verify=expected,unsupported -fsyntax-only %s

#if (defined(__loongarch__) || defined(__x86_64__)) && !__has_attribute(model)
#error "Should support model attribute"
#endif

int a __attribute((model("tiny")));    // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'tiny' is not supported on this target}} \
                                       // x86_64-error {{code model 'tiny' is not supported on this target}}
int b __attribute((model("small")));   // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'small' is not supported on this target}}
int c __attribute((model("normal")));  // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-error {{code model 'normal' is not supported on this target}}
int d __attribute((model("kernel")));  // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'kernel' is not supported on this target}} \
                                       // x86_64-error {{code model 'kernel' is not supported on this target}}
int e __attribute((model("medium")));  // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-error {{code model 'medium' is not supported on this target}}
int f __attribute((model("large")));   // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'large' is not supported on this target}}
int g __attribute((model("extreme"))); // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-error {{code model 'extreme' is not supported on this target}}

void __attribute((model("extreme"))) h() {} // unsupported-warning {{unknown attribute 'model' ignored}} \
                                            // loongarch64-error {{'model' attribute only applies to non-TLS global variables}} \
                                            // x86_64-error {{'model' attribute only applies to non-TLS global variables}}

// NVPTX doesn't support thread_local at all.
#ifndef __NVPTX__
thread_local
#endif
int i __attribute((model("extreme"))); // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{'model' attribute only applies to non-TLS global variables}} \
                                       // x86_64-error {{'model' attribute only applies to non-TLS global variables}}
