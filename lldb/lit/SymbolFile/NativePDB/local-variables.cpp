// clang-format off

// REQUIRES: system-windows
// RUN: %build -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/local-variables.lldbinit 2>&1 | FileCheck %s

int Function(int Param1, char Param2) {
  unsigned Local1 = Param1 + 1;
  char Local2 = Param2 + 1;
  ++Local1;
  ++Local2;
  return Local1;
}

int main(int argc, char **argv) {
  int SomeLocal = argc * 2;
  return Function(SomeLocal, 'a');
}

// CHECK:      (lldb) target create "{{.*}}local-variables.cpp.tmp.exe"
// CHECK-NEXT: Current executable set to '{{.*}}local-variables.cpp.tmp.exe'
// CHECK-NEXT: (lldb) command source -s 0 '{{.*}}local-variables.lldbinit'
// CHECK-NEXT: Executing commands in '{{.*}}local-variables.lldbinit'.
// CHECK-NEXT: (lldb) break set -n main
// CHECK-NEXT: Breakpoint 1: where = local-variables.cpp.tmp.exe`main + {{.*}} at local-variables.cpp:{{.*}}, address = {{.*}}
// CHECK-NEXT: (lldb) run a b c d e f g
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = breakpoint 1.1
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`main(argc=8, argv={{.*}}) at local-variables.cpp:{{.*}}
// CHECK-NEXT:    14   }
// CHECK-NEXT:    15
// CHECK-NEXT:    16   int main(int argc, char **argv) {
// CHECK-NEXT: -> 17     int SomeLocal = argc * 2;
// CHECK-NEXT:    18     return Function(SomeLocal, 'a');
// CHECK-NEXT:    19   }
// CHECK-NEXT:    20

// CHECK:      Process {{.*}} launched: '{{.*}}local-variables.cpp.tmp.exe'
// CHECK-NEXT: (lldb) p argc
// CHECK-NEXT: (int) $0 = 8
// CHECK-NEXT: (lldb) step
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = step in
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`main(argc=8, argv={{.*}}) at local-variables.cpp:{{.*}}
// CHECK-NEXT:    15
// CHECK-NEXT:    16 int main(int argc, char **argv) {
// CHECK-NEXT:    17     int SomeLocal = argc * 2;
// CHECK-NEXT: -> 18     return Function(SomeLocal, 'a');
// CHECK-NEXT:    19 }
// CHECK-NEXT:    20

// CHECK:      (lldb) p SomeLocal
// CHECK-NEXT: (int) $1 = 16
// CHECK-NEXT: (lldb) step
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = step in
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-NEXT:    6
// CHECK-NEXT:    7
// CHECK-NEXT:    8 int Function(int Param1, char Param2) {
// CHECK-NEXT: -> 9      unsigned Local1 = Param1 + 1;
// CHECK-NEXT:    10     char Local2 = Param2 + 1;
// CHECK-NEXT:    11     ++Local1;
// CHECK-NEXT:    12     ++Local2;

// CHECK:      (lldb) p Param1
// CHECK-NEXT: (int) $2 = 16
// CHECK-NEXT: (lldb) p Param2
// CHECK-NEXT: (char) $3 = 'a'
// CHECK-NEXT: (lldb) step
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = step in
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-NEXT:    7
// CHECK-NEXT:    8    int Function(int Param1, char Param2) {
// CHECK-NEXT:    9      unsigned Local1 = Param1 + 1;
// CHECK-NEXT: -> 10     char Local2 = Param2 + 1;
// CHECK-NEXT:    11     ++Local1;
// CHECK-NEXT:    12     ++Local2;
// CHECK-NEXT:    13     return Local1;

// CHECK:      (lldb) p Param1
// CHECK-NEXT: (int) $4 = 16
// CHECK-NEXT: (lldb) p Param2
// CHECK-NEXT: (char) $5 = 'a'
// CHECK-NEXT: (lldb) p Local1
// CHECK-NEXT: (unsigned int) $6 = 17
// CHECK-NEXT: (lldb) step
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = step in
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-NEXT:    8    int Function(int Param1, char Param2) {
// CHECK-NEXT:    9      unsigned Local1 = Param1 + 1;
// CHECK-NEXT:    10     char Local2 = Param2 + 1;
// CHECK-NEXT: -> 11     ++Local1;
// CHECK-NEXT:    12     ++Local2;
// CHECK-NEXT:    13     return Local1;
// CHECK-NEXT:    14   }

// CHECK:      (lldb) p Param1
// CHECK-NEXT: (int) $7 = 16
// CHECK-NEXT: (lldb) p Param2
// CHECK-NEXT: (char) $8 = 'a'
// CHECK-NEXT: (lldb) p Local1
// CHECK-NEXT: (unsigned int) $9 = 17
// CHECK-NEXT: (lldb) p Local2
// CHECK-NEXT: (char) $10 = 'b'
// CHECK-NEXT: (lldb) step
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = step in
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-NEXT:    9      unsigned Local1 = Param1 + 1;
// CHECK-NEXT:    10     char Local2 = Param2 + 1;
// CHECK-NEXT:    11     ++Local1;
// CHECK-NEXT: -> 12     ++Local2;
// CHECK-NEXT:    13     return Local1;
// CHECK-NEXT:    14   }
// CHECK-NEXT:    15

// CHECK:      (lldb) p Param1
// CHECK-NEXT: (int) $11 = 16
// CHECK-NEXT: (lldb) p Param2
// CHECK-NEXT: (char) $12 = 'a'
// CHECK-NEXT: (lldb) p Local1
// CHECK-NEXT: (unsigned int) $13 = 18
// CHECK-NEXT: (lldb) p Local2
// CHECK-NEXT: (char) $14 = 'b'
// CHECK-NEXT: (lldb) step
// CHECK-NEXT: Process {{.*}} stopped
// CHECK-NEXT: * thread #1, stop reason = step in
// CHECK-NEXT:     frame #0: {{.*}} local-variables.cpp.tmp.exe`Function(Param1=16, Param2='a') at local-variables.cpp:{{.*}}
// CHECK-NEXT:    10      char Local2 = Param2 + 1;
// CHECK-NEXT:    11     ++Local1;
// CHECK-NEXT:    12     ++Local2;
// CHECK-NEXT: -> 13     return Local1;
// CHECK-NEXT:    14   }
// CHECK-NEXT:    15
// CHECK-NEXT:    16   int main(int argc, char **argv) {

// CHECK:      (lldb) p Param1
// CHECK-NEXT: (int) $15 = 16
// CHECK-NEXT: (lldb) p Param2
// CHECK-NEXT: (char) $16 = 'a'
// CHECK-NEXT: (lldb) p Local1
// CHECK-NEXT: (unsigned int) $17 = 18
// CHECK-NEXT: (lldb) p Local2
// CHECK-NEXT: (char) $18 = 'c'
// CHECK-NEXT: (lldb) continue
// CHECK-NEXT: Process {{.*}} resuming
// CHECK-NEXT: Process {{.*}} exited with status = 18 (0x00000012)

// CHECK:      (lldb) target modules dump ast
// CHECK-NEXT: Dumping clang ast for {{.*}} modules.
// CHECK-NEXT: TranslationUnitDecl
// CHECK-NEXT: |-FunctionDecl {{.*}} main 'int (int, char **)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} argc 'int'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} argv 'char **'
// CHECK-NEXT: `-FunctionDecl {{.*}} Function 'int (int, char)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} Param1 'int'
// CHECK-NEXT:   `-ParmVarDecl {{.*}} Param2 'char'
