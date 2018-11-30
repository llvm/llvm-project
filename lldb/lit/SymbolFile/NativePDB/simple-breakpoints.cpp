// clang-format off
// REQUIRES: lld

// Test that we can set simple breakpoints using PDB on any platform.
// RUN: %clang_cl /Z7 /GS- /GR- /c /Fo%t.obj -- %s 
// RUN: lld-link /DEBUG /nodefaultlib /entry:main /OUT:%t.exe /PDB:%t.pdb -- %t.obj
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/breakpoints.lldbinit | FileCheck %s

// Use different indentation style for each overload so that the starting
// line is in a different place.
int OvlGlobalFn(int X) {
  return X + 42;
}

int OvlGlobalFn(int X, int Y) { return X + Y + 42; }

int OvlGlobalFn(int X, int Y, int Z)
{
  return X + Y + Z + 42;
}

static int StaticFn(int X) {
  return X + 42;
}

int main(int argc, char **argv) {
  // Make sure they don't get optimized out.
  // Note the comments here, we want to make sure the line number reported
  // for the breakpoint is the first actual line of code.
  int Result = OvlGlobalFn(argc) + OvlGlobalFn(argc, argc)
    + OvlGlobalFn(argc, argc, argc) + StaticFn(argc);
  return Result;
}


// CHECK:      (lldb) target create "{{.*}}simple-breakpoints.cpp.tmp.exe"
// CHECK:      Current executable set to '{{.*}}simple-breakpoints.cpp.tmp.exe'
// CHECK:      (lldb) break set -n main
// CHECK:      Breakpoint 1: where = simple-breakpoints.cpp.tmp.exe`main + {{[0-9]+}}
// CHECK-SAME:               at simple-breakpoints.cpp:31
// CHECK:      (lldb) break set -n OvlGlobalFn
// CHECK:      Breakpoint 2: 3 locations.
// CHECK:      (lldb) break set -n StaticFn
// CHECK:      Breakpoint 3: where = simple-breakpoints.cpp.tmp.exe`StaticFn + {{[0-9]+}}
// CHECK-SAME:               at simple-breakpoints.cpp:24
// CHECK:      (lldb) break set -n DoesntExist
// CHECK:      Breakpoint 4: no locations (pending).
// CHECK:      (lldb) break list
// CHECK:      Current breakpoints:
// CHECK:      1: name = 'main', locations = 1
// CHECK:        1.1: where = simple-breakpoints.cpp.tmp.exe`main + {{[0-9]+}}
// CHECK-SAME:                at simple-breakpoints.cpp:31
// CHECK:      2: name = 'OvlGlobalFn', locations = 3
// CHECK:        2.1: where = simple-breakpoints.cpp.tmp.exe`OvlGlobalFn + {{[0-9]+}}
// CHECK-SAME:                at simple-breakpoints.cpp:13
// CHECK:        2.2: where = simple-breakpoints.cpp.tmp.exe`OvlGlobalFn
// CHECK-SAME:                at simple-breakpoints.cpp:16
// CHECK:        2.3: where = simple-breakpoints.cpp.tmp.exe`OvlGlobalFn + {{[0-9]+}}
// CHECK-SAME:                at simple-breakpoints.cpp:20
// CHECK:      3: name = 'StaticFn', locations = 1
// CHECK:        3.1: where = simple-breakpoints.cpp.tmp.exe`StaticFn + {{[0-9]+}}
// CHECK-SAME:                at simple-breakpoints.cpp:24
// CHECK:      4: name = 'DoesntExist', locations = 0 (pending)
