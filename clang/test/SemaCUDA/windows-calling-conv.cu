// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple \
// RUN:   x86_64-pc-windows-msvc -fms-compatibility -fcuda-is-device \
// RUN:   -fsyntax-only -verify -x hip %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-compatibility \
// RUN:   -fsyntax-only -verify -x hip %s

// expected-no-diagnostics

typedef void (__stdcall* funcTy)();
void invoke(funcTy f);

static void __stdcall callee() noexcept {
}

void foo() {
   invoke(callee);
}
