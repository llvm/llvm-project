// RUN: %clang_cc1 -triple aarch64 -verify=expected,aarch64 -fsyntax-only %s
// RUN: %clang_cc1 -triple loongarch64 -verify=expected,loongarch64 -fsyntax-only %s
// RUN: %clang_cc1 -triple mips64 -verify=expected,mips64 -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64 -verify=expected,powerpc64 -fsyntax-only %s
// RUN: %clang_cc1 -triple riscv64 -verify=expected,riscv64 -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64 -verify=expected,x86_64 -fsyntax-only %s

#if defined(__loongarch__) && !__has_attribute(model)
#error "Should support model attribute"
#endif

int a __attribute((model("tiny")));    // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'tiny' is not supported on this target}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}
int b __attribute((model("small")));   // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'small' is not supported on this target}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}
int c __attribute((model("normal")));  // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}
int d __attribute((model("kernel")));  // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'kernel' is not supported on this target}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}
int e __attribute((model("medium")));  // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}
int f __attribute((model("large")));   // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'large' is not supported on this target}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}
int g __attribute((model("extreme"))); // aarch64-warning {{unknown attribute 'model' ignored}} \
                                       // mips64-warning {{unknown attribute 'model' ignored}} \
                                       // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                       // riscv64-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-warning {{unknown attribute 'model' ignored}}

void __attribute((model("extreme"))) h() {} // aarch64-warning {{unknown attribute 'model' ignored}} \
                                            // loongarch64-error {{'model' attribute only applies to non-TLS global variables}} \
                                            // mips64-warning {{unknown attribute 'model' ignored}} \
                                            // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                            // riscv64-warning {{unknown attribute 'model' ignored}} \
                                            // x86_64-warning {{unknown attribute 'model' ignored}}

thread_local int i __attribute((model("extreme"))); // aarch64-warning {{unknown attribute 'model' ignored}} \
                                                    // loongarch64-error {{'model' attribute only applies to non-TLS global variables}} \
                                                    // mips64-warning {{unknown attribute 'model' ignored}} \
                                                    // powerpc64-warning {{unknown attribute 'model' ignored}} \
                                                    // riscv64-warning {{unknown attribute 'model' ignored}} \
                                                    // x86_64-warning {{unknown attribute 'model' ignored}}
