// RUN: c-index-test core -print-source-symbols -- %s | FileCheck %s

// Linkage decls are skipped in USRs for enclosed items.
// Linkage decls themselves don't have USRs (no lines between ns and X).
// CHECK: {{[0-9]+}}:11 | namespace/C++ | ns | c:@N@ns |
// CHECK-NEXT: {{[0-9]+}}:33 | variable/C | X | c:@N@ns@X |
namespace ns { extern "C" { int X; } }
