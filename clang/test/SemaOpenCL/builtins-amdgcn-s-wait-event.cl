// xUN: %clang_cc1 -fsyntax-only -triple amdgcn-- -target-cpu gfx1100 -verify=ALL,GFX11 %s
// RUN: %clang_cc1 -fsyntax-only -triple amdgcn-- -target-cpu gfx1200 -verify=ALL,GFX12 %s

void test(int x) {
  // ALL-error@+1 {{argument to '__builtin_amdgcn_s_wait_event' must be a constant integer}}
  __builtin_amdgcn_s_wait_event(x);

  // GFX11-expected-no-diagnostics
  // GFX12-warning@+2 {{event mask has no effect for target}}
  // GFX12-note@+1 {{value of 2 valid for export_ready for gfx11 and gfx12+}}
  __builtin_amdgcn_s_wait_event(0); // 0 does nothing on gfx12

  // GFX11-expected-no-diagnostics
  // GFX12-warning@+2 {{event mask has no effect for target}}
  // GFX12-note@+1 {{value of 2 valid for export_ready for gfx11 and gfx12+}}
  __builtin_amdgcn_s_wait_event(1); // 1 does nothing on gfx11

  __builtin_amdgcn_s_wait_event(2); // expected-no-diagnostics

  // ALL-warning@+2 {{event mask has no effect for target}}
  // ALL-note@+1 {{value of 2 valid for export_ready for gfx11 and gfx12+}}
  __builtin_amdgcn_s_wait_event(3);

  // ALL-warning@+2 {{event mask has no effect for target}}
  // ALL-note@+1 {{value of 2 valid for export_ready for gfx11 and gfx12+}}
  __builtin_amdgcn_s_wait_event(-1);
}
