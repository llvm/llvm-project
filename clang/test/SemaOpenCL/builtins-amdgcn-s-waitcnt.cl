// RUN: %clang_cc1 -fsyntax-only -triple amdgpu11.00-- -verify=GFX11 %s
// RUN: %clang_cc1 -fsyntax-only -triple amdgpu12.00-- -verify=GFX1200 %s
// RUN: %clang_cc1 -fsyntax-only -triple amdgpu12.50-- -verify=GFX1250 %s

void test(void) {
  // GFX11-no-diagnostics
  // GFX1200-no-diagnostics
  // GFX1250-error@+1 {{builtin is not supported on this target}}
  __builtin_amdgcn_s_waitcnt(0);
}
