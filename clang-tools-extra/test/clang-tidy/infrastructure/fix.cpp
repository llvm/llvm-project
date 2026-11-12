// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,google-explicit-constructor,llvm-namespace-comment' -fix -export-fixes=%t.yaml -- > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s
// RUN: FileCheck -input-file=%t.yaml -check-prefix=CHECK-YAML %s

namespace i {
void f(); // So that the namespace isn't empty.
}
// CHECK: } // namespace i
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-YAML: ReplacementText: ' // namespace i'

class A { A(int i); };
// CHECK: class A { explicit A(int i); };
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-MESSAGES: clang-tidy applied 2 of 2 suggested fixes.
// CHECK-YAML: ReplacementText: 'explicit '
