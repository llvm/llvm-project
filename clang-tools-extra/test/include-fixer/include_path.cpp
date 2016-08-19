// RUN: mkdir -p %T/include-fixer/include
// RUN: mkdir -p %T/include-fixer/symbols
// RUN: mkdir -p %T/include-fixer/build
// RUN: mkdir -p %T/include-fixer/src
// RUN: sed 's|test_dir|%/T/include-fixer|g' %S/Inputs/database_template.json > %T/include-fixer/build/compile_commands.json
// RUN: echo -e '#include "bar.h"\nb::a::bar f;' > %T/include-fixer/src/bar.cpp
// RUN: echo 'namespace b { namespace a { class bar {}; } }' > %T/include-fixer/include/bar.h
// RUN: cd %T/include-fixer/build
// RUN: find-all-symbols -output-dir=%T/include-fixer/symbols -p=. %T/include-fixer/src/bar.cpp
// RUN: find-all-symbols -merge-dir=%T/include-fixer/symbols %T/include-fixer/build/find_all_symbols.yaml
// RUN: FileCheck -input-file=%T/include-fixer/build/find_all_symbols.yaml -check-prefix=CHECK-YAML %s
//
// RUN: echo 'b::a::bar f;' > %T/include-fixer/src/bar.cpp
// RUN: clang-include-fixer -db=yaml -input=%T/include-fixer/build/find_all_symbols.yaml -minimize-paths=true -p=. %T/include-fixer/src/bar.cpp
// RUN: FileCheck -input-file=%T/include-fixer/src/bar.cpp %s

// CHECK-YAML: ..{{[/\\]}}include{{[/\\]}}bar.h
// CHECK: #include "bar.h"
// CHECK: b::a::bar f;
