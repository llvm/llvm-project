// Checks that -cc1 command lines generated for inputs from the Standard Library
// module manifest are correctly adjusted, specifically that:
// - manifest-specified local arguments are applied to the corresponding entry.
// - "-Wno-reserved-module-identifier" is added for Standard library modules.

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

// RUN: %clang -std=c++20 \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -stdlib=libc++ \
// RUN:   -resource-dir=%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck %s -check-prefix=MAIN-CC1 -check-prefix=STDLIB-MOD-CC1 -DPREFIX=%/t

// MAIN-CC1: "-cc1"
// MAIN-CC1-SAME: "main.cpp"
// MAIN-CC1-NOT: "-Wno-reserved-module-identifier"
// MAIN-CC1-NOT: "-internal-isystem" "[[PREFIX]]/Inputs/usr/lib/x86_64-linux-gnu/../share/libc++/v1/"

// STDLIB-MOD-CC1: "-cc1"
// STDLIB-MOD-CC1-SAME: "[[PREFIX]]/Inputs/usr/lib/x86_64-linux-gnu/../share/libc++/v1/std.cppm"
// STDLIB-MOD-CC1-SAME: "-Wno-reserved-module-identifier"
// STDLIB-MOD-CC1-SAME: "-internal-isystem" "[[PREFIX]]/Inputs/usr/lib/x86_64-linux-gnu/../share/libc++/v1/"

//--- main.cpp
import std;

int main() {}

//--- std.cppm
export module std;

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
          "../share/libc++/v1/"
         ]
      }
    }
  ]
}
