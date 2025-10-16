// This test checks the on-demand scanning of system inputs, in particular:
// 1. Inputs for unused system modules are not scanned.
// 2. Imports between system modules are supported and scanned on demand.

// RUN: split-file %s %t

// The standard library modules manifest (libc++.modules.json) is discovered
// relative to the installed C++ standard library runtime libraries
// We need to create them in order for Clang to find the manifest.
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: mkdir -p %t/Inputs/usr/lib/x86_64-linux-gnu
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.so
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.a

// RUN: sed "s|DIR|%/t|g" %t/libc++.modules.json.in > \
// RUN:   %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.modules.json

// RUN: mkdir -p %t/Inputs/usr/lib/share/libc++/v1
// RUN: mkdir -p %t/Inputs/usr/lib/share/libc++/v2
// RUN: cat %t/std.cppm > %t/Inputs/usr/lib/share/libc++/v1/std.cppm
// RUN: cat %t/std.compat.cppm > %t/Inputs/usr/lib/share/libc++/v1/std.compat.cppm

// RUN: %clang -std=c++20 \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -stdlib=libc++ \
// RUN:   -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp %t/foo.cpp \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t

// CHECK:      remark: using standard modules manifest file '[[PREFIX]]/Inputs/usr/lib/x86_64-linux-gnu/libc++.modules.json' [-Rmodules-driver]
// CHECK:      remark: printing module dependency graph [-Rmodules-driver]
// CHECK-NEXT: digraph "Module Dependency Graph" {
// CHECK-NEXT:         label="Module Dependency Graph";

// CHECK:              "[[PREFIX]]/main.cpp" [ fillcolor=[[COLOR1:[0-9]+]],label="{ Kind: Non-module | Filename: [[PREFIX]]/main.cpp }"];
// CHECK-NEXT:         "[[PREFIX]]/foo.cpp" [ fillcolor=[[COLOR1]],label="{ Kind: Non-module | Filename: [[PREFIX]]/foo.cpp }"];
// CHECK-NEXT:         "std" [ fillcolor=[[COLOR2:[0-9]+]],label="{ Kind: C++ named module | Module name: std | Filename: [[PREFIX]]/Inputs/usr/lib/share/libc++/v1/std.cppm | Input origin: System | Hash: {{.*}} }"];
// CHECK-NEXT:         "std.compat" [ fillcolor=[[COLOR2]],label="{ Kind: C++ named module | Module name: std.compat | Filename: [[PREFIX]]/Inputs/usr/lib/share/libc++/v1/std.compat.cppm | Input origin: System | Hash: {{.*}} }"];
// CHECK-NEXT:         "core" [ fillcolor=[[COLOR2]],label="{ Kind: C++ named module | Module name: core | Filename: [[PREFIX]]/core.cxxm | Input origin: System | Hash: {{.*}} }"];

// CHECK:              "[[PREFIX]]/main.cpp" -> "std";
// CHECK-NEXT:         "[[PREFIX]]/main.cpp" -> "std.compat";
// CHECK-NEXT:         "[[PREFIX]]/foo.cpp" -> "core";
// CHECK-NEXT:         "std.compat" -> "std";
// CHECK-NEXT: }


//--- main.cpp
import std;
import std.compat;

//--- foo.cpp
import core;

//--- std.cppm
export module std;

//--- std.compat.cppm
export module std.compat;
import std;

// The module 'core' is isn't really a system module in libc++ or libstdc++.
// This is only to test that any module marked with '"is-std-library": true' can 
// be imported on demand.
//--- core.cxxm
export module core;

//--- unused.cppm
export module unused;

//--- libc++.modules.json.in
{
  "version": 1,
  "revision": 1,
  "modules": [
    {
      "logical-name": "std",
      "source-path": "../share/libc++/v1/std.cppm",
      "is-std-library": true
    },
    {
      "logical-name": "std.compat",
      "source-path": "../share/libc++/v1/std.compat.cppm",
      "is-std-library": true
    },
    {
      "logical-name": "core",
      "source-path": "DIR/core.cxxm",
      "is-std-library": true
    },
    {
      "logical-name": "unused",
      "source-path": "DIR/unused.cppm",
      "is-std-library": true
    }
  ]
}
