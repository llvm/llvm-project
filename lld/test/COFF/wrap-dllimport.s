// REQUIRES: x86

// Check that we can wrap a dllimported symbol, so that references to
// __imp_<symbol> gets redirected to a symbol that already exists or a defined
// local import instead.

// RUN: split-file %s %t.dir
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %t.dir/main.s -o %t.main.obj
// RUN: llvm-mc -filetype=obj -triple=i686-win32-gnu %t.dir/other.s -o %t.other.obj

// RUN: lld-link -dll -out:%t.dll %t.other.obj -noentry -safeseh:no -export:foo -export:bar -implib:%t.lib
// RUN: lld-link -out:%t.exe %t.main.obj %t.lib -entry:entry -subsystem:console -debug:symtab -safeseh:no -wrap:foo -wrap:bar -lldmap:%t.map
// RUN: llvm-objdump -s -d --print-imm-hex %t.exe | FileCheck %s

// CHECK:      Contents of section .rdata:
// CHECK-NEXT:  402000 0c104000

// CHECK:      Disassembly of section .text:
// CHECK-EMPTY:
// CHECK:      00401000 <_entry>:
// CHECK-NEXT:   401000: ff 25 00 20 40 00             jmpl    *0x402000
// CHECK-NEXT:   401006: ff 25 00 00 00 00             jmpl    *0x0
// CHECK-EMPTY:
// CHECK-NEXT: 0040100c <___wrap_foo>:
// CHECK-NEXT:   40100c: c3                            retl
// CHECK-EMPTY:
// CHECK-NEXT: 0040100d <___wrap_bar>:
// CHECK-NEXT:   40100d: c3                            retl

// The first jmpl instruction in _entry points at an address in 0x402000,
// which is the first 4 bytes of the .rdata section (above), which is a
// pointer that points at ___wrap_foo.

// The second jmpl instruction in _entry points to null because the referenced
// symbol `__imp____wrap_bar` is declared as a weak reference to prevent pull a
// reference from an external DLL.

#--- main.s
.global _entry
_entry:
  jmpl *__imp__foo
  jmpl *__imp__bar

.global ___wrap_foo
___wrap_foo:
  ret

.weak __imp____wrap_bar
.global ___wrap_bar
___wrap_bar:
  ret

#--- other.s
.global _foo
_foo:
  ret

.global _bar
_bar:
  ret
