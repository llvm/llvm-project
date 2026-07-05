These are debug info integration tests similar to the ones in the parent
directory, except that these are designed to test compatibility between clang,
lld, and cdb, the command line debugger that ships as part of the Microsoft
Windows SDK. The debugger command language that cdb uses is very different from
gdb and LLDB, so it's useful to be able to write some tests directly in the cdb
command language.

An example header for a CDB test, of which there are currently none:

// RUN: %clang_cl %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK
