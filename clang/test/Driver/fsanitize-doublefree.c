// RUN: %clang --target=x86_64-linux-gnu -fsanitize=doublefree %s -### 2>&1 | FileCheck %s --check-prefix=DSAN
// DSAN: "-fsanitize=doublefree"
// DSAN: libclang_rt.dsan

// RUN: %clang --target=x86_64-linux-gnu -fsanitize=doublefree,undefined %s -### 2>&1 | FileCheck %s --check-prefix=DSAN-UBSAN
// DSAN-UBSAN: libclang_rt.dsan
// DSAN-UBSAN: libclang_rt.ubsan_standalone

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=doublefree,leak %s -fsyntax-only 2>&1 | FileCheck %s --check-prefix=DSAN-LEAK
// DSAN-LEAK: '-fsanitize=doublefree' not allowed with '-fsanitize=leak'

// RUN: not %clang --target=x86_64-linux-gnu -fsanitize=doublefree,scudo %s -fsyntax-only 2>&1 | FileCheck %s --check-prefix=DSAN-SCUDO
// DSAN-SCUDO: '-fsanitize=doublefree' not allowed with '-fsanitize=scudo'

// RUN: not %clang --target=x86_64-unknown-freebsd -fsanitize=doublefree %s -fsyntax-only 2>&1 | FileCheck %s --check-prefix=FREEBSD
// FREEBSD: unsupported option '-fsanitize=doublefree' for target 'x86_64-unknown-freebsd'

// RUN: not %clang --target=wasm32-unknown-emscripten -fsanitize=doublefree %s -fsyntax-only 2>&1 | FileCheck %s --check-prefix=EMSCRIPTEN
// EMSCRIPTEN: unsupported option '-fsanitize=doublefree' for target 'wasm32-unknown-emscripten'

int main(void) { return 0; }
