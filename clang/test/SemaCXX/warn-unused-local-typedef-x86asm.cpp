// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -std=c++11 -Wunused-local-typedef -verify -fasm-blocks %s
// expected-no-diagnostics
void use_in_asm() {
  typedef struct {
    int a;
    int b;
  } A;
  __asm mov eax, [eax].A.b

  using Alias = struct {
    int a;
    int b;
  };
  __asm mov eax, [eax].Alias.b
}
