// RUN: mkdir -p %t.dir/clang-include-fixer/include
// RUN: mkdir -p %t.dir/clang-include-fixer/symbols
// RUN: mkdir -p %t.dir/clang-include-fixer/build
// RUN: mkdir -p %t.dir/clang-include-fixer/src
// RUN: sed 's|test_dir|%/t.dir/clang-include-fixer|g' %S/Inputs/database_template.json > %t.dir/clang-include-fixer/build/compile_commands.json
// RUN: echo -e '#include "bar.h"\nb::a::bar f;' > %t.dir/clang-include-fixer/src/bar.cpp
// RUN: echo 'namespace b { namespace a { class bar {}; } }' > %t.dir/clang-include-fixer/include/bar.h
// RUN: cd %t.dir/clang-include-fixer/build
// RUN: find-all-symbols -output-dir=%t.dir/clang-include-fixer/symbols -p=. %t.dir/clang-include-fixer/src/bar.cpp
// RUN: find-all-symbols -merge-dir=%t.dir/clang-include-fixer/symbols %t.dir/clang-include-fixer/build/find_all_symbols.yaml
// RUN: FileCheck -input-file=%t.dir/clang-include-fixer/build/find_all_symbols.yaml -check-prefix=CHECK-YAML %s
//
// RUN: echo 'b::a::bar f;' > %t.dir/clang-include-fixer/src/bar.cpp
// RUN: clang-include-fixer -db=yaml -input=%t.dir/clang-include-fixer/build/find_all_symbols.yaml -minimize-paths=true -p=. %t.dir/clang-include-fixer/src/bar.cpp
// RUN: FileCheck -input-file=%t.dir/clang-include-fixer/src/bar.cpp %s

// CHECK-YAML: ..{{[/\\]}}include{{[/\\]}}bar.h
// CHECK: #include "bar.h"
// CHECK: b::a::bar f;
