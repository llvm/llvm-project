// RUN: c-index-test -test-load-source-reparse 2 all %s -Xclang -add-plugin -Xclang clang-include-fixer -fspell-checking -Xclang -plugin-arg-clang-include-fixer -Xclang -input=%p/Inputs/fake_yaml_db.yaml 2>&1 | FileCheck %s

foo f;
foo g;
unknown u;

// CHECK: yamldb_plugin.cpp:3:1: error: unknown type name 'foo'; did you mean 'foo'?
// CHECK: Number FIX-ITs = 1
// CHECK: FIX-IT: Replace [3:1 - 3:4] with "foo"
// CHECK: yamldb_plugin.cpp:3:1: note: Add '#include "foo.h"' to provide the missing declaration [clang-include-fixer]
// CHECK: Number FIX-ITs = 1
// CHECK: FIX-IT: Insert "#include "foo.h"
// CHECK: yamldb_plugin.cpp:4:1: error: unknown type name 'foo'; did you mean 'foo'?
// CHECK: Number FIX-ITs = 1
// CHECK: FIX-IT: Replace [4:1 - 4:4] with "foo"
// CHECK: yamldb_plugin.cpp:4:1: note: Add '#include "foo.h"' to provide the missing declaration [clang-include-fixer]
// CHECK: Number FIX-ITs = 1
// CHECK: FIX-IT: Insert "#include "foo.h"
// CHECK: " at 3:1
// CHECK: yamldb_plugin.cpp:5:1:
// CHECK: error: unknown type name 'unknown'
// CHECK: Number FIX-ITs = 0
// CHECK-NOT: error
// CHECK-NOT: FIX-IT
