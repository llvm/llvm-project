// RUN: %clang_cc1 -fsyntax-only -Wunknown-pragmas -verify %s
// RUN: %clang_cc1 -E %s 2>&1 | FileCheck --strict-whitespace %s

// GCC doesn't expand macro args for unrecognized pragmas.
#define bar xX
#pragma foo bar   // expected-warning {{unknown pragma ignored}}
// CHECK-NOT: unknown pragma in STDC namespace
// CHECK: {{^}}#pragma foo bar{{$}}

#pragma STDC FP_CONTRACT ON
#pragma STDC FP_CONTRACT OFF
#pragma STDC FP_CONTRACT DEFAULT
#pragma STDC FP_CONTRACT IN_BETWEEN  // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
// CHECK: {{^}}#pragma STDC FP_CONTRACT ON{{$}}
// CHECK: {{^}}#pragma STDC FP_CONTRACT OFF{{$}}
// CHECK: {{^}}#pragma STDC FP_CONTRACT DEFAULT{{$}}
// CHECK: {{^}}#pragma STDC FP_CONTRACT IN_BETWEEN{{$}}

#pragma STDC FENV_ACCESS ON          // expected-warning {{pragma STDC FENV_ACCESS ON is not supported, ignoring pragma}}
#pragma STDC FENV_ACCESS OFF
#pragma STDC FENV_ACCESS DEFAULT
#pragma STDC FENV_ACCESS IN_BETWEEN   // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
// CHECK: {{^}}#pragma STDC FENV_ACCESS ON{{$}}
// CHECK: {{^}}#pragma STDC FENV_ACCESS OFF{{$}}
// CHECK: {{^}}#pragma STDC FENV_ACCESS DEFAULT{{$}}
// CHECK: {{^}}#pragma STDC FENV_ACCESS IN_BETWEEN{{$}}

#pragma STDC CX_LIMITED_RANGE ON
#pragma STDC CX_LIMITED_RANGE OFF
#pragma STDC CX_LIMITED_RANGE DEFAULT 
#pragma STDC CX_LIMITED_RANGE IN_BETWEEN   // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
// CHECK: {{^}}#pragma STDC CX_LIMITED_RANGE ON{{$}}
// CHECK: {{^}}#pragma STDC CX_LIMITED_RANGE OFF{{$}}
// CHECK: {{^}}#pragma STDC CX_LIMITED_RANGE DEFAULT{{$}}
// CHECK: {{^}}#pragma STDC CX_LIMITED_RANGE IN_BETWEEN{{$}}

#pragma STDC CX_LIMITED_RANGE    // expected-warning {{expected 'ON' or 'OFF' or 'DEFAULT' in pragma}}
#pragma STDC CX_LIMITED_RANGE ON FULL POWER  // expected-warning {{expected end of directive in pragma}}
// CHECK: {{^}}#pragma STDC CX_LIMITED_RANGE{{$}}
// CHECK: {{^}}#pragma STDC CX_LIMITED_RANGE ON FULL POWER{{$}}

#pragma STDC SO_GREAT  // expected-warning {{unknown pragma in STDC namespace}}
#pragma STDC   // expected-warning {{unknown pragma in STDC namespace}}
// CHECK: {{^}}#pragma STDC SO_GREAT{{$}}
// CHECK: {{^}}#pragma STDC{{$}}
