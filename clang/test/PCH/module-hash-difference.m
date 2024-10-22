// RUN: rm -rf %t.mcp
// RUN: rm -rf %t.err
// RUN: %clang_cc1 -emit-pch -o %t.pch %s -I %S/Inputs/modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp
// RUN: not %clang_cc1 -fsyntax-only -include-pch %t.pch %s -I %S/Inputs/modules -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash 2> %t.err
// RUN: FileCheck -input-file=%t.err %s

// CHECK: error: AST file '{{.*}}' was compiled with module cache path {{.*}}, but the path is currently {{.*}}
@import Foo;
