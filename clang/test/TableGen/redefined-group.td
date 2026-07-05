// RUN: clang-tblgen -gen-clang-diag-groups -I%S %s -o /dev/null 2>&1 | FileCheck %s
include "DiagnosticBase.inc"

def NamedGroup : DiagGroup<"a">;
def InNamedGroup1 : Warning<"">, InGroup<DiagGroup<"a">>;
def InNamedGroup2 : Warning<"">, InGroup  < DiagGroup<"a"> >;
// CHECK: redefined-group.td:[[@LINE-3]]:5: error: group 'a' is defined more than once
// CHECK: redefined-group.td:[[@LINE-3]]:5: note: also implicitly defined here
// CHECK: redefined-group.td:[[@LINE-3]]:5: note: also implicitly defined here

def : DiagGroup<"b">;
def InUnnamedGroup : Warning<"">, InGroup<DiagGroup<"b">>;
// CHECK: redefined-group.td:[[@LINE-2]]:1: error: group 'b' is defined more than once
// CHECK: redefined-group.td:[[@LINE-2]]:5: note: also implicitly defined here

def ImplicitGroup1 : Warning<"">, InGroup<DiagGroup<"c">>;
def ImplicitGroup2 : Warning<"">, InGroup<DiagGroup<"c">>;
def ImplicitGroup3 : Warning<"">,
  InGroup<DiagGroup<"c">>;
// CHECK: redefined-group.td:[[@LINE-4]]:5: error: group 'c' is implicitly defined more than once
// CHECK: redefined-group.td:[[@LINE-4]]:5: note: also implicitly defined here
// CHECK: redefined-group.td:[[@LINE-4]]:5: note: also implicitly defined here

def NamedAndUnnamed : DiagGroup<"d">;
def : DiagGroup<"d">;
// CHECK: redefined-group.td:[[@LINE-2]]:5: error: group 'd' is defined more than once
// CHECK: redefined-group.td:[[@LINE-2]]:1: note: also defined here

def : DiagGroup<"e">;
def NamedAndUnnamed2 : DiagGroup<"e">;
// CHECK: redefined-group.td:[[@LINE-1]]:5: error: group 'e' is defined more than once
// CHECK: redefined-group.td:[[@LINE-3]]:1: note: also defined here

def InGroupF1 : Warning<"">, InGroup<DiagGroup<"f">>;
def : DiagGroup<"f">; // FIXME: It'd be nice to also note this, but it's hard to detect.
def InGroupF2 : Warning<"">, InGroup<DiagGroup<"f">>;
def GroupF : DiagGroup<"f">;
def InGroupF3 : Warning<"">, InGroup<GroupF>;
def InGroupF4 : Warning<"">, InGroup<DiagGroup<"f">>;
// CHECK: redefined-group.td:[[@LINE-5]]:1: error: group 'f' is defined more than once
// CHECK: redefined-group.td:[[@LINE-7]]:5: note: also implicitly defined here
// CHECK: redefined-group.td:[[@LINE-6]]:5: note: also implicitly defined here
// CHECK: redefined-group.td:[[@LINE-4]]:5: note: also implicitly defined here
