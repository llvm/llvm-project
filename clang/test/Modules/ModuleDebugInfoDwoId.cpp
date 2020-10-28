// Tests that dwoIds in modules match the dwoIDs in the main file.

// REQUIRES: asserts

// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -triple %itanium_abi_triple -x objective-c++ -std=c++11 -debugger-tuning=lldb -debug-info-kind=limited -fmodules -fmodule-format=obj -dwarf-ext-refs -fimplicit-module-maps -fmodules-cache-path=%t.cache %s -I %S/Inputs -emit-llvm -o %t.ll -mllvm -debug-only=pchcontainer 2> %t.mod-out
// RUN: cat %t.ll %t.mod-out | FileCheck %s
// RUN: cat %t.ll | FileCheck --check-prefix=CHECK-REALIDS %s
// RUN: cat %t.mod-out | FileCheck --check-prefix=CHECK-REALIDS %s

@import DebugDwoId;

Dummy d;

// Find the emitted dwoID for DebugInfoId and compare it against the one in the PCM.
// CHECK: DebugDwoId-{{[A-Z0-9]+}}.pcm
// CHECK-SAME: dwoId: [[DWOID:[0-9]+]]
// CHECK: dwoId: [[DWOID]]
// CHECK-NEXT: !DIFile(filename: "DebugDwoId"

// Make sure the dwo IDs are real IDs and not fallback values (~1ULL).
// CHECK-REALIDS-NOT: dwoId: 18446744073709551615
