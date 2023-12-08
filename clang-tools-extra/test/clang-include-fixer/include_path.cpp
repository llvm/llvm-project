// RUN: mkdir -p %T/clang-include-fixer/include
// RUN: mkdir -p %T/clang-include-fixer/symbols
// RUN: mkdir -p %T/clang-include-fixer/build
// RUN: mkdir -p %T/clang-include-fixer/src
// RUN: sed 's|test_dir|%/T/clang-include-fixer|g' %S/Inputs/database_template.json > %T/clang-include-fixer/build/compile_commands.json
// RUN: echo -e '#include "bar.h"\nb::a::bar f;' > %T/clang-include-fixer/src/bar.cpp
// RUN: echo 'namespace b { namespace a { class bar {}; } }' > %T/clang-include-fixer/include/bar.h
// RUN: cd %T/clang-include-fixer/build
// RUN: find-all-symbols -output-dir=%T/clang-include-fixer/symbols -p=. %T/clang-include-fixer/src/bar.cpp
// RUN: find-all-symbols -merge-dir=%T/clang-include-fixer/symbols %T/clang-include-fixer/build/find_all_symbols.yaml
// RUN: FileCheck -input-file=%T/clang-include-fixer/build/find_all_symbols.yaml -check-prefix=CHECK-YAML %s
//
// RUN: echo 'b::a::bar f;' > %T/clang-include-fixer/src/bar.cpp
// RUN: clang-include-fixer -db=yaml -input=%T/clang-include-fixer/build/find_all_symbols.yaml -minimize-paths=true -p=. %T/clang-include-fixer/src/bar.cpp
// RUN: FileCheck -input-file=%T/clang-include-fixer/src/bar.cpp %s

// CHECK-YAML: ..{{[/\\]}}include{{[/\\]}}bar.h
// CHECK: #include "bar.h"
// CHECK: b::a::bar f;
