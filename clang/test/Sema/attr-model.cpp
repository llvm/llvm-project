// RUN: %clang_cc1 -triple aarch64 -verify=unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple loongarch64 -verify=loongarch64 -fsyntax-only %s
// RUN: %clang_cc1 -triple mips64 -verify=unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc64 -verify=unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple riscv64 -verify=unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64 -verify=x86_64 -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx64-unknown-cuda -fcuda-is-device -x cuda -verify=ignored -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgcn -verify=ignored -fsyntax-only %s
// RUN: %clang_cc1 -triple r600 -verify=ignored -fsyntax-only %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -verify=ignored -fsyntax-only %s
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -verify=ignored -fsyntax-only %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -verify=ignored -fsyntax-only %s

// RUN: %clang_cc1 -triple x86_64 -aux-triple nvptx64 -x cuda -verify=x86_64 -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx64 -aux-triple x86_64 -x cuda -fcuda-is-device -verify=nvptx64-x86_64 -fsyntax-only %s
// RUN: %clang_cc1 -triple aarch64 -aux-triple nvptx64 -x cuda -verify=unsupported -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx64 -aux-triple aarch64 -x cuda -fcuda-is-device -verify=nvptx64-unsupported -fsyntax-only %s

#if (defined(__loongarch__) || defined(__x86_64__)) && !__has_attribute(model)
#error "Should support model attribute"
#endif

int a __attribute((model("tiny")));    // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'tiny' is not supported on this target}} \
                                       // x86_64-error {{code model 'tiny' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // nvptx64-x86_64-error {{code model 'tiny' is not supported on this target}}
int b __attribute((model("small")));   // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'small' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}}
int c __attribute((model("normal")));  // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-error {{code model 'normal' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // nvptx64-x86_64-error {{code model 'normal' is not supported on this target}}
int d __attribute((model("kernel")));  // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'kernel' is not supported on this target}} \
                                       // x86_64-error {{code model 'kernel' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // nvptx64-x86_64-error {{code model 'kernel' is not supported on this target}}
int e __attribute((model("medium")));  // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-error {{code model 'medium' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // nvptx64-x86_64-error {{code model 'medium' is not supported on this target}}
int f __attribute((model("large")));   // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{code model 'large' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}}
int g __attribute((model("extreme"))); // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // x86_64-error {{code model 'extreme' is not supported on this target}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // nvptx64-x86_64-error {{code model 'extreme' is not supported on this target}}

void __attribute((model("extreme"))) h() {} // unsupported-warning {{unknown attribute 'model' ignored}} \
                                            // ignored-error {{'model' attribute only applies to non-TLS global variables}} \
                                            // loongarch64-error {{'model' attribute only applies to non-TLS global variables}} \
                                            // x86_64-error {{'model' attribute only applies to non-TLS global variables}} \
                                            // nvptx64-unsupported-error {{'model' attribute only applies to non-TLS global variables}} \
                                            // nvptx64-x86_64-error {{'model' attribute only applies to non-TLS global variables}}

#if !defined(__CUDA__) || !defined(__CUDA_ARCH__)
// if we are compiling for non-cuda host, or host mode in a CUDA compile
#if !defined(__AMDGCN__) && !defined(__R600__) && !defined(__SPIRV__)
// for all non-cuda hosts, above targets don't support thread_local
thread_local
#endif
#endif
int i __attribute((model("extreme"))); // unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // loongarch64-error {{'model' attribute only applies to non-TLS global variables}} \
                                       // x86_64-error {{'model' attribute only applies to non-TLS global variables}} \
                                       // nvptx64-unsupported-warning {{unknown attribute 'model' ignored}} \
                                       // nvptx64-x86_64-error {{code model 'extreme' is not supported on this target}}
