// RUN: clang-tblgen -gen-clang-diag-groups -I%S %s -o /dev/null 2>&1 | FileCheck --strict-whitespace %s
include "DiagnosticBase.inc"

// Do not move this line; it is referred to by absolute line number in the
// FileCheck lines below.
def NamedGroup : DiagGroup<"name">;


def InNamedGroup : Warning<"">, InGroup<DiagGroup<"name">>;
//      CHECK: anonymous-groups.td:[[@LINE-1]]:41: error: group 'name' is referred to anonymously
// CHECK-NEXT: {{^def InNamedGroup : Warning<"">, InGroup<DiagGroup<"name">>;}}
// CHECK-NEXT: {{^                                ~~~~~~~~\^~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT: {{^                                InGroup<NamedGroup>}}
// CHECK-NEXT: anonymous-groups.td:6:1: note: group defined here
// CHECK-NEXT: def NamedGroup : DiagGroup<"name">;
// CHECK-NEXT: ^


def AlsoInNamedGroup : Warning<"">, InGroup  < DiagGroup<"name"> >;
//      CHECK: anonymous-groups.td:[[@LINE-1]]:48: error: group 'name' is referred to anonymously
// CHECK-NEXT: {{^def AlsoInNamedGroup : Warning<"">, InGroup  < DiagGroup<"name"> >;}}
// CHECK-NEXT: {{^                                    ~~~~~~~~~~~\^~~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT: {{^                                    InGroup<NamedGroup>}}
// CHECK-NEXT: anonymous-groups.td:6:1: note: group defined here
// CHECK-NEXT: def NamedGroup : DiagGroup<"name">;
// CHECK-NEXT: ^


def AnonymousGroup : Warning<"">, InGroup<DiagGroup<"anonymous">>;
def AlsoAnonymousGroup : Warning<"">, InGroup<DiagGroup<"anonymous">>;
def AnonymousGroupAgain : Warning<"">,
  InGroup<DiagGroup<"anonymous">>;

//      CHECK: anonymous-groups.td:[[@LINE-5]]:43: error: group 'anonymous' is referred to anonymously
// CHECK-NEXT: {{^def AnonymousGroup : Warning<"">, InGroup<DiagGroup<"anonymous">>;}}
// CHECK-NEXT: {{^                                  ~~~~~~~~\^~~~~~~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT: anonymous-groups.td:[[@LINE-7]]:47: note: also referenced here
// CHECK-NEXT: {{^def AlsoAnonymousGroup : Warning<"">, InGroup<DiagGroup<"anonymous">>;}}
// CHECK-NEXT: {{^                                      ~~~~~~~~\^~~~~~~~~~~~~~~~~~~~~~~}}
// CHECK-NEXT: anonymous-groups.td:[[@LINE-8]]:11: note: also referenced here
// CHECK-NEXT: {{^  InGroup<DiagGroup<"anonymous">>;}}
// CHECK-NEXT: {{^  ~~~~~~~~\^~~~~~~~~~~~~~~~~~~~~~~}}
