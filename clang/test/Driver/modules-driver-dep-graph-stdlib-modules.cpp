// This test checks the on-demand scanning of system inputs, in particular:
// 1. System inputs are scanned only when needed.
// 2. Imports between system modules are supported.

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
// RUN: cat %t/std.cppm > %t/Inputs/usr/lib/share/libc++/v1/std.cppm
// RUN: cat %t/std.compat.cppm > %t/Inputs/usr/lib/share/libc++/v1/std.compat.cppm
// RUN: cat %t/core.cppm > %t/Inputs/usr/lib/share/core.cppm
// RUN: cat %t/unused.cppm > %t/Inputs/usr/lib/share/unused.cppm

// RUN: %clang -std=c++20		    \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -stdlib=libc++ \
// RUN:   -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp %t/foo.cpp \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t

// CHECK:      digraph "Module Dependency Graph" {
//
// CHECK:              "[[PREFIX]]/main.cpp-x86_64-unknown-linux-gnu" [fillcolor=3, label="{ Filename: [[PREFIX]]/main.cpp | Triple: x86_64-unknown-linux-gnu }"];
// CHECK-NEXT:         "[[PREFIX]]/foo.cpp-x86_64-unknown-linux-gnu" [fillcolor=3, label="{ Filename: [[PREFIX]]/foo.cpp | Triple: x86_64-unknown-linux-gnu }"];
// CHECK-NEXT:         "std-x86_64-unknown-linux-gnu" [fillcolor=2, label="{ Filename: {{.*}} | Module type: Named module | Module name: std | Triple: x86_64-unknown-linux-gnu }"];
// CHECK-NEXT:         "std.compat-x86_64-unknown-linux-gnu" [fillcolor=2, label="{ Filename: {{.*}} | Module type: Named module | Module name: std.compat | Triple: x86_64-unknown-linux-gnu }"];
// CHECK-NEXT:         "core-x86_64-unknown-linux-gnu" [fillcolor=2, label="{ Filename: {{.*}} | Module type: Named module | Module name: core | Triple: x86_64-unknown-linux-gnu }"];
//
// CHECK:              "std-x86_64-unknown-linux-gnu" -> "[[PREFIX]]/main.cpp-x86_64-unknown-linux-gnu";
// CHECK-NEXT:         "std-x86_64-unknown-linux-gnu" -> "[[PREFIX]]/foo.cpp-x86_64-unknown-linux-gnu";
// CHECK-NEXT:         "std-x86_64-unknown-linux-gnu" -> "std.compat-x86_64-unknown-linux-gnu";
// CHECK-NEXT:         "std.compat-x86_64-unknown-linux-gnu" -> "[[PREFIX]]/main.cpp-x86_64-unknown-linux-gnu";
// CHECK-NEXT:         "core-x86_64-unknown-linux-gnu" -> "[[PREFIX]]/foo.cpp-x86_64-unknown-linux-gnu";
// CHECK-NEXT: }


//--- main.cpp
import std;
import std.compat;

//--- foo.cpp
import std;
import core;

//--- std.cppm
export module std;

//--- std.compat.cppm
export module std.compat;
import std;

// The module 'core' is isn't really a system module in libc++ or libstdc++.
// This is only to test that any module marked with '"is-std-library": true' can 
// be imported on demand.
//--- core.cppm
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
      "source-path": "../share/core.cppm",
      "is-std-library": true
    },
    {
      "logical-name": "unused",
      "source-path": "../share/unused.cppm",
      "is-std-library": true
    }
  ]
}
