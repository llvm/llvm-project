// Verify that a malformed standard modules manifest triggers an error.

// RUN: split-file %s %t

// The standard library modules manifest (libc++.modules.json) is discovered
// relative to the installed C++ standard library runtime libraries
// We need to create them in order for Clang to find the manifest.
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: mkdir -p %t/Inputs/usr/lib/x86_64-linux-gnu
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.so
// RUN: touch %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.a

// Add the standard module manifest itself.
// RUN: cat %t/libc++.modules.json.in > \
// RUN:   %t/Inputs/usr/lib/x86_64-linux-gnu/libc++.modules.json

// RUN: not %clang -std=c++20 \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -stdlib=libc++ \
// RUN:   -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -DPREFIX=%/t

// CHECK: remark: using standard modules manifest file '[[PREFIX]]/Inputs/usr/lib/x86_64-linux-gnu/libc++.modules.json' [-Rmodules-driver]
// CHECK: error: failure while parsing standard modules manifest: '[5:7, byte=57]: Invalid JSON value (true?)'

//--- main.cpp
// empty

//--- libc++.modules.json.in
{
  "version": 1,
  "revision": 1,
  "modules": [
     this is malformed.
  ]
}
