// RUN: %clang %s -### 2>&1 | FileCheck -check-prefix NO_MODULE_CACHE %s
// RUN: %clang -fmodule-cache-path blarg %s -### 2>&1 | FileCheck -check-prefix WITH_MODULE_CACHE %s

// CHECK-NO_MODULE_CACHE: {{clang.*"-fmodule-cache-path"}}

// CHECK-WITH_MODULE_CACHE: {{clang.*"-fmodule-cache-path" "blarg"}}
