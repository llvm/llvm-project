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
// RUN:   -resource-dir=%t/FakeSysroot/usr/lib/x86_64-linux-gnu \
// RUN:   --sysroot=%t/FakeSysroot \
// RUN:   -L%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -c %t/main.cpp 2>&1 \
// RUN:   --verbose \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t %s

// Skip past diagnostics omitted during the dependency scan.
// CHECK: clang: remark: printing module dependency graph [-Rmodules-driver]

// Checks that the standard library modules are primary outputs.
// TODO: Test the same for -fno-modules-reduced-bmi when supported by 
// -fmodules-driver.
// CHECK: -o {{.*std-[^ ]*\.pcm}}
// CHECK-SAME: -emit-reduced-module-interface
// CHECK-SAME: -main-file-name std.cppm
// CHECK: -o {{.*std\.compat-[^ ]*\.pcm}}
// CHECK-SAME: -emit-reduced-module-interface
// CHECK-SAME: -main-file-name std.compat.cppm

// Checks that the object file is generated for the main TU.
// CHECK: -emit-obj
// CHECK-SAME: -main-file-name main.cpp

// Checks that the standard library modules are successfully imported.
// CHECK: [[PREFIX]]/main.cpp:1:1: remark: importing module 'std.compat' from
// CHECK: [[PREFIX]]/main.cpp:1:1: remark: importing module 'std' into 'std.compat' from
// CHECK: [[PREFIX]]/main.cpp:2:1: remark: importing module 'std' from

// Checks that standard library modules are still precompiled with -emit-llvm.
// RUN: %clang -std=c++23 -stdlib=libc++ \
// RUN:   -fmodules-driver -Rmodules-driver -Rmodule-import \
// RUN:   -resource-dir=%t/FakeSysroot/usr/lib/x86_64-linux-gnu \
// RUN:   --sysroot=%t/FakeSysroot \
// RUN:   -L%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   %t/main.cpp -S -emit-llvm -o out.ll \
// RUN:   --verbose 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefix=CHECK-EMIT-LLVM %s

// Skip past diagnostics omitted during the dependency scan.
// CHECK-EMIT-LLVM: clang: remark: printing module dependency graph [-Rmodules-driver]

// CHECK-EMIT-LLVM: -o {{.*std-[^ ]*\.pcm}}
// CHECK-EMIT-LLVM-SAME: -emit-reduced-module-interface
// CHECK-EMIT-LLVM-SAME: -main-file-name std.cppm
// CHECK-EMIT-LLVM: -o {{.*std\.compat-[^ ]*\.pcm}}
// CHECK-EMIT-LLVM-SAME: -emit-reduced-module-interface
// CHECK-EMIT-LLVM-SAME: -main-file-name std.compat.cppm
// CHECK-EMIT-LLVM: -emit-llvm
// CHECK-EMIT-LLVM-SAME: -main-file-name main.cpp

// Checks that standard library modules are still precompiled with -fsyntax-only.
// RUN: %clang -std=c++23 -stdlib=libc++ \
// RUN:   -fmodules-driver -Rmodules-driver -Rmodule-import \
// RUN:   -resource-dir=%t/FakeSysroot/usr/lib/x86_64-linux-gnu \
// RUN:   --sysroot=%t/FakeSysroot \
// RUN:   -L%t/Inputs/usr/lib/x86_64-linux-gnu \
// RUN:   --target=x86_64-linux-gnu \
// RUN:   -fsyntax-only \
// RUN:   %t/main.cpp \
// RUN:   --verbose 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefix=CHECK-SYNTAX-ONLY %s

// Skip past diagnostics omitted during the dependency scan.
// CHECK-SYNTAX-ONLY: clang: remark: printing module dependency graph [-Rmodules-driver]

// CHECK-SYNTAX-ONLY: -o {{.*std-[^ ]*\.pcm}}
// CHECK-SYNTAX-ONLY-SAME: -emit-reduced-module-interface
// CHECK-SYNTAX-ONLY-SAME: -main-file-name std.cppm
// CHECK-SYNTAX-ONLY: -o {{.*std\.compat-[^ ]*\.pcm}}
// CHECK-SYNTAX-ONLY-SAME: -emit-reduced-module-interface
// CHECK-SYNTAX-ONLY-SAME: -main-file-name std.compat.cppm

// CHECK-SYNTAX-ONLY: -cc1 {{.*}} -Rmodules-driver
// CHECK-SYNTAX-ONLY-NOT: {{ -o }}
