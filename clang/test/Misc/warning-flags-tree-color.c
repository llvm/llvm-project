// REQUIRES: ansi-escape-sequences

// RUN: diagtool tree --color -Wgnu | FileCheck -check-prefix CHECK-COLOR %s
// RUN: diagtool tree --color=always -Wgnu | FileCheck -check-prefix CHECK-COLOR %s
// RUN: diagtool tree --internal --color=always -Wgnu | FileCheck -check-prefix CHECK-COLOR %s
// RUN: diagtool tree --color=always --internal -Wgnu | FileCheck -check-prefix CHECK-COLOR %s
// RUN: diagtool tree --color=auto -Wgnu | FileCheck -check-prefix CHECK-NOCOLOR %s
// RUN: diagtool tree --color=never -Wgnu | FileCheck -check-prefix CHECK-NOCOLOR %s
// RUN: not diagtool tree --color=invalid -Wgnu 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

// CHECK-COLOR: {{.}}[0;32mGREEN{{.}}[0m = enabled by default
// CHECK-COLOR: {{.}}[0;33mYELLOW{{.}}[0m = disabled by default
// CHECK-COLOR: {{.}}[0;31mRED{{.}}[0m = unimplemented (accepted for GCC compatibility)
// CHECK-COLOR: {{.}}[0;33m-Wgnu

// CHECK-NOCOLOR: GREEN = enabled by default
// CHECK-NOCOLOR: YELLOW = disabled by default
// CHECK-NOCOLOR: RED = unimplemented (accepted for GCC compatibility)
// CHECK-NOCOLOR: -Wgnu

// CHECK-INVALID: Invalid value for --color: 'invalid'; expected 'always', 'auto', or 'never'
// CHECK-INVALID: Usage: diagtool tree [--internal] [--color[=<always|auto|never>]] [<diagnostic-group>]
