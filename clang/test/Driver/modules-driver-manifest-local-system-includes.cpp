// Verifies that the local system include directories listed in the standard
// module manifest are added to corresponding driver jobs.

// RUN: split-file %s %t

// The standard library modules manifest (libc++.modules.json) is discovered
// relative to the installed C++ standard library runtime libraries
// We need to create them in order for Clang to find the manifest.
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: mkdir -p %t/Inputs/usr/lib/x86_64-linux-gnu
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.so
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.a

// RUN: cat %t/libc++.modules.json.in > \
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
// RUN:   %t/main.cpp \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t

// CHECK: "-cc1" {{.*}} "-main-file-name" "std.cppm" {{.*}} "-internal-isystem" "[[PREFIX]]/Inputs/usr/lib/share/libc++/v1"
// CHECK: "-cc1" {{.*}} "-main-file-name" "std.compat.cppm" {{.*}} "-internal-isystem" "[[PREFIX]]/Inputs/usr/lib/share/libc++/v1" "-internal-isystem" "[[PREFIX]]/Inputs/usr/lib/share/libc++/v2"

//--- main.cpp
// Import the system modules to ensure that the corresponding jobs don't get
// deleted.
import std;
import std.compat;

//--- std.cppm
export module std;

//--- std.compat.cppm
export module std.compat;
import std;

//--- libc++.modules.json.in
{
  "version": 1,
  "revision": 1,
  "modules": [
    {
      "logical-name": "std",
      "source-path": "../share/libc++/v1/std.cppm",
      "is-std-library": true,
      "local-arguments": {
        "system-include-directories": [
          "../share/libc++/v1"
        ]
      }
    },
    {
      "logical-name": "std.compat",
      "source-path": "../share/libc++/v1/std.compat.cppm",
      "is-std-library": true,
      "local-arguments": {
        "system-include-directories": [
	  "../share/libc++/v1",
          "../share/libc++/v2"
        ]
      }
    }
  ]
}

