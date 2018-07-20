// RUN: clang-tblgen -gen-clang-diag-groups -I%S %s -o /dev/null 2>&1 | FileCheck --strict-whitespace %s
include "DiagnosticBase.inc"

def NamedGroup : DiagGroup<"name">;

def InNamedGroup : Warning<"">, InGroup<DiagGroup<"name">>;
//      CHECK: tg-fixits.td:[[@LINE-1]]:1: error: group 'name' is referred to anonymously
// CHECK-NEXT: {{^def InNamedGroup : Warning<"">, InGroup<DiagGroup<"name">>;}}

def Wrapped : Warning<"">, InGroup<DiagGroup<
  "name">>;
//      CHECK: tg-fixits.td:[[@LINE-2]]:1: error: group 'name' is referred to anonymously
// CHECK-NEXT: {{^def Wrapped : Warning<"">, InGroup<DiagGroup<}}

def AlsoWrapped : Warning<"">, InGroup<
  DiagGroup<"name">>;
//      CHECK: tg-fixits.td:[[@LINE-2]]:1: error: group 'name' is referred to anonymously

// The following line has Unicode characters in it; do not change them!
// FIXME: For now, we just give up on printing carets/ranges/fixits for
// lines with Unicode in them, because SMDiagnostic don't keep a byte<->column
// map around to line things up like Clang does.
def Unicode : Warning<"ユニコード">, InGroup<DiagGroup<"name">>;
//      CHECK: tg-fixits.td:[[@LINE-1]]:1: error: group 'name' is referred to anonymously
// CHECK-NEXT: def Unicode : Warning<"{{[^"]+}}">, InGroup<DiagGroup<"name">>;
