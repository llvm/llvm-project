// RUN: mkdir -p %T/clang-doc/build
// RUN: mkdir -p %T/clang-doc/include
// RUN: mkdir -p %T/clang-doc/src
// RUN: mkdir -p %T/clang-doc/docs
// RUN: sed 's|$test_dir|%/T/clang-doc|g' %S/Inputs/clang-doc-project1/database_template.json > %T/clang-doc/build/compile_commands.json
// RUN: cp %S/Inputs/clang-doc-project1/*.h  %T/clang-doc/include
// RUN: cp %S/Inputs/clang-doc-project1/*.cpp %T/clang-doc/src
// RUN: cd %T/clang-doc/build
// RUN: clang-doc --format=html --executor=all-TUs --output=%T/clang-doc/docs ./compile_commands.json


