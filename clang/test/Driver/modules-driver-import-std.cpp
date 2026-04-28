// Checks that -fmodules-driver correctly handles the import of Standard library
// modules.

// REQUIRES: x86-registered-target

// The standard library modules manifest (libc++.modules.json) is discovered
// relative to the installed C++ standard library runtime libraries
// We need to create them in order for Clang to find the manifest.
// RUN: rm -rf %t && split-file %s %t
// RUN: mkdir -p %t/FakeSysroot/usr/lib/x86_64-linux-gnu
// RUN: touch %t/FakeSysroot/usr/lib/x86_64-linux-gnu/libc++.so
// RUN: touch %t/FakeSysroot/usr/lib/x86_64-linux-gnu/libc++.a

// RUN: sed "s|DIR|%/t|g" %t/libc++.modules.json.in > \
// RUN:   %t/FakeSysroot/usr/lib/x86_64-linux-gnu/libc++.modules.json

// RUN: mkdir -p %t/FakeSysroot/usr/lib/share/libc++/v1
// RUN: cat %t/std.cppm > %t/FakeSysroot/usr/lib/share/libc++/v1/std.cppm
// RUN: cat %t/std.compat.cppm > %t/FakeSysroot/usr/lib/share/libc++/v1/std.compat.cppm

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
    }
  ]
}

//--- std.cppm
export module std;

//--- std.compat.cppm
export module std.compat;
import std;

//--- main.cpp
import std.compat;
import std;

int main() {}

// RUN: %clang -std=c++23 -stdlib=libc++ \
// RUN:   -fmodules-driver -Rmodules-driver -Rmodule-import \
// RUN:   -stdlib=libc++ \
// RUN:   -resource-dir=%t/FakeSysroot/usr/lib/x86_64-linux-gnu \
// RUN:   --sysroot=%t/FakeSysroot \
// RUN:   -L%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -c %t/main.cpp 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t %s

// CHECK: [[PREFIX]]/main.cpp:1:1: remark: importing module 'std.compat' from
// CHECK: [[PREFIX]]/main.cpp:1:1: remark: importing module 'std' into 'std.compat' from
// CHECK: [[PREFIX]]/main.cpp:2:1: remark: importing module 'std' from
