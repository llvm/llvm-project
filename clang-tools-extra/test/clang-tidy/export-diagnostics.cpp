// RUN: grep -Ev "// *[A-Z-]+:" %s > %t-input.cpp
// RUN: clang-tidy %t-input.cpp -checks='-*,google-explicit-constructor,clang-diagnostic-missing-prototypes' -export-fixes=%t.yaml -- -Wmissing-prototypes > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s -implicit-check-not='{{warning|error|note}}:'
// RUN: FileCheck -input-file=%t.yaml -check-prefix=CHECK-YAML %s
#define X(n) void n ## n() {}
X(f)

// CHECK-MESSAGES: -input.cpp:2:1: warning: no previous prototype for function 'ff' [clang-diagnostic-missing-prototypes]
// CHECK-MESSAGES: -input.cpp:1:19: note: expanded from macro 'X'
// CHECK-MESSAGES: {{^}}note: expanded from here{{$}}

// CHECK-YAML: ---
// CHECK-YAML-NEXT: MainSourceFile:  '{{.*}}-input.cpp'
// CHECK-YAML-NEXT: Diagnostics:
// CHECK-YAML-NEXT:   - DiagnosticName:  clang-diagnostic-missing-prototypes
// CHECK-YAML-NEXT:     Message:         'no previous prototype for function ''ff'''
// CHECK-YAML-NEXT:     FileOffset:      30
// CHECK-YAML-NEXT:     FilePath:        '{{.*}}-input.cpp'
// CHECK-YAML-NEXT:     Notes:
// CHECK-YAML-NEXT:       - Message:         'expanded from macro ''X'''
// CHECK-YAML-NEXT:         FilePath:        '{{.*}}-input.cpp'
// CHECK-YAML-NEXT:         FileOffset:      18
// CHECK-YAML-NEXT:       - Message:         expanded from here
// CHECK-YAML-NEXT:         FilePath:        ''
// CHECK-YAML-NEXT:         FileOffset:      0
// CHECK-YAML-NEXT:     Replacements:    []
// CHECK-YAML-NEXT: ...

