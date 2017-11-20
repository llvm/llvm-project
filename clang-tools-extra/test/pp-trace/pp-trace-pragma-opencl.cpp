// RUN: pp-trace -ignore FileChanged,MacroDefined %s -x cl | FileCheck --strict-whitespace %s

#pragma OPENCL EXTENSION all : disable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : disable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// CHECK: ---
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:3:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaOpenCLExtension
// CHECK-NEXT:   NameLoc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:3:26"
// CHECK-NEXT:   Name: all
// CHECK-NEXT:   StateLoc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:3:32"
// CHECK-NEXT:   State: 0
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:4:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaOpenCLExtension
// CHECK-NEXT:   NameLoc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:4:26"
// CHECK-NEXT:   Name: cl_khr_int64_base_atomics
// CHECK-NEXT:   StateLoc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:4:54"
// CHECK-NEXT:   State: 0
// CHECK-NEXT: - Callback: PragmaDirective
// CHECK-NEXT:   Loc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:5:1"
// CHECK-NEXT:   Introducer: PIK_HashPragma
// CHECK-NEXT: - Callback: PragmaOpenCLExtension
// CHECK-NEXT:   NameLoc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:5:26"
// CHECK-NEXT:   Name: cl_khr_int64_base_atomics
// CHECK-NEXT:   StateLoc: "{{.*}}{{[/\\]}}pp-trace-pragma-opencl.cpp:5:54"
// CHECK-NEXT:   State: 1
// CHECK-NEXT: - Callback: EndOfMainFile
// CHECK-NEXT: ...
