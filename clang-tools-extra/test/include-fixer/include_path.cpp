// REQUIRES: shell
// RUN: mkdir -p %T/include-fixer/include
// RUN: mkdir -p %T/include-fixer/build
// RUN: mkdir -p %T/include-fixer/src
// RUN: sed 's|test_dir|%T/include-fixer|g' %S/Inputs/database_template.json > %T/include-fixer/build/compile_commands.json
// RUN: cp %S/Inputs/fake_yaml_db.yaml %T/include-fixer/build/fake_yaml_db.yaml
// RUN: echo 'b::a::bar f;' > %T/include-fixer/src/bar.cpp
// RUN: touch %T/include-fixer/include/bar.h
// RUN: cd %T/include-fixer/build
// RUN: clang-include-fixer -db=yaml -input=fake_yaml_db.yaml -p=. %T/include-fixer/src/bar.cpp
// RUN: FileCheck -input-file=%T/include-fixer/src/bar.cpp %s

// CHECK: #include "bar.h"
// CHECK: b::a::bar f;
