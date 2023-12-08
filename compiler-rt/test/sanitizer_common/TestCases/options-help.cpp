// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=help=1,include_if_exists=___some_path_that_does_not_exist___  %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=help=1,symbolize=0 %run %t 2>&1 | FileCheck --check-prefix=CHECK-CV %s
// RUN: %env_tool_opts=help=1,sancov_path=/long/path/that/requires/truncation/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaB \
// RUN:   %run %t 2>&1 | FileCheck --check-prefix=CHECK-TRUNCATION %s

int main() {
}

// CHECK: Available flags for {{.*}}Sanitizer:

//
// Bool option
// CHECK: {{^[ \t]+symbolize$}}
// CHECK-NEXT: (Current Value: true)
//
// String option
// CHECK: {{^[ \t]+log_path$}}
// CHECK-NEXT: (Current Value: {{.+}})
//
// int option
// CHECK: {{^[ \t]+verbosity$}}
// CHECK-NEXT: (Current Value: {{-?[0-9]+}})
//
// HandleSignalMode option
// CHECK: {{^[ \t]+handle_segv$}}
// CHECK-NEXT: (Current Value: {{0|1|2}})
//
// uptr option
// CHECK: {{^[ \t]+mmap_limit_mb$}}
// CHECK-NEXT: (Current Value: 0x{{[0-9a-fA-F]+}})
//
// FlagHandlerInclude option
// CHECK: include_if_exists
// CHECK-NEXT: (Current Value: ___some_path_that_does_not_exist___)

// Test we show the current value and not the default.
// CHECK-CV: {{^[ \t]+symbolize$}}
// CHECK-CV-NEXT: (Current Value: false)

// Test truncation of long paths.
// CHECK-TRUNCATION: sancov_path
// CHECK-TRUNCATION-NEXT: (Current Value Truncated: /long/path/that/requires/truncation/aaa{{a+}})
