// under test:
// - parsing and using compile_commands
// - export fixes to yaml file

// use %t as directory instead of file,
// because "compile_commands.json" must have exactly that name:
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '[{"directory":"%/S","command":"clang++ -c %s","file":"%s"}]' \
// RUN:      > %t/compile_commands.json

// execute and check:
// RUN: cd "%t"
// RUN: %run_clang_tidy -checks='-*,bugprone-sizeof-container,modernize-use-auto' \
// RUN:                 -p="%/t" -export-fixes=%t/fixes.yaml > %t/msg.txt 2>&1
// RUN: FileCheck -input-file=%t/msg.txt -check-prefix=CHECK-MESSAGES %s \
// RUN:           -implicit-check-not='{{warning|error|note}}:'
// RUN: FileCheck -input-file=%t/fixes.yaml -check-prefix=CHECK-YAML %s

#include <vector>
int main()
{
  std::vector<int> vec;
  std::vector<int>::iterator iter = vec.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-YAML: modernize-use-auto
  
  return sizeof(vec);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: sizeof() doesn't return the size of the container; did you mean .size()? [bugprone-sizeof-container]
  // CHECK-YAML: bugprone-sizeof-container
  // After https://reviews.llvm.org/D72730 --> CHECK-YAML-NOT: bugprone-sizeof-container
}

