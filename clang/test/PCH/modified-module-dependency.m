// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: echo '@import test;' > %t-dir/prefix.h
// RUN: echo 'void foo(void);' > %t-dir/test.h
// RUN: cp %S/modified-module-dependency.module.map %t-dir/module.map

// Precompile prefix.pch.
// RUN: %clang_cc1 -x objective-c -I %t-dir -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-dir/cache -fdisable-module-hash -emit-pch %t-dir/prefix.h -o %t-dir/prefix.pch

// Modify the dependency.
// RUN: echo ' ' >> %t-dir/test.h

// Run and check the diagnostics.
// RUN: not %clang_cc1 -x objective-c -I %t-dir -include-pch %t-dir/prefix.pch -fmodules -fimplicit-module-maps -fmodules-cache-path=%t-dir/cache -fdisable-module-hash -fsyntax-only %s 2> %t-dir/log
// RUN: FileCheck %s < %t-dir/log

// CHECK: file '[[TEST_H:.*[/\\]test\.h]]' has been modified since the precompiled header '[[PREFIX_PCH:.*/prefix\.pch]]' was built
// CHECK: '[[TEST_H]]' required by '[[TEST_PCM:.*[/\\]test\.pcm]]'
// CHECK: '[[TEST_PCM]]' required by '[[PREFIX_PCH]]'
// CHECK: please rebuild precompiled header '[[PREFIX_PCH]]'
