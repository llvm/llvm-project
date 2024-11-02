// RUN: env HOME=%t.home %clang -fmodules -fimplicit-module-maps %s -### 2>&1 | FileCheck -check-prefix CHECK-NO_MODULE_CACHE %s
// RUN: %clang -fmodules -fimplicit-module-maps -fmodules-cache-path=blarg %s -### 2>&1 | FileCheck -check-prefix CHECK-WITH_MODULE_CACHE %s

// CHECK-NO_MODULE_CACHE: {{clang.*"-fmodules-cache-path=.*ModuleCache"}}

// CHECK-WITH_MODULE_CACHE: {{clang.*"-fmodules-cache-path=blarg"}}
