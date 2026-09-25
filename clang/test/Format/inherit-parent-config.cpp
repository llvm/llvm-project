// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/code

// RUN: cd %t.dir
// RUN: echo "BasedOnStyle: Google" > .clang-format
// RUN: echo "IndentWidth: 4" >> .clang-format

// RUN: cd code
// RUN: echo "BasedOnStyle: InheritParentConfig" > .clang-format
// RUN: echo "---" >> .clang-format
// RUN: echo "Language: Cpp" >> .clang-format

// RUN: clang-format -style=file:.clang-format %s \
// RUN:   | FileCheck %s --strict-whitespace
// CHECK: {{^ {8}//}}

s = R"CPP(
    void foo() {
  // "IndentWidth: 4" applies here, resulting in 8 leading spaces.
    }
)CPP";
