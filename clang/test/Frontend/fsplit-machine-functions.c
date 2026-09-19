/// -cc1 accepts every late function splitting mode and rejects unknown ones.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsplit-machine-functions=none -emit-llvm -o /dev/null %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsplit-machine-functions=bbsections -emit-llvm -o /dev/null %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsplit-machine-functions=all -emit-llvm -o /dev/null %s
/// The legacy spelling is an alias for '=all'.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsplit-machine-functions -emit-llvm -o /dev/null %s

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsplit-machine-functions=bogus -emit-llvm -o /dev/null %s 2>&1 | FileCheck %s
// CHECK: invalid value 'bogus' in '-fsplit-machine-functions=bogus'

void f(void) {}
