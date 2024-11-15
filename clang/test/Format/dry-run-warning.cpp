// RUN: echo '{' > %t.json
// RUN: echo '  "married": true' >> %t.json
// RUN: echo '}' >> %t.json

// RUN: clang-format -n -style=LLVM %t.json 2>&1 | FileCheck %s -allow-empty

// RUN: clang-format -n -style=LLVM < %t.json 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK2 -strict-whitespace

// RUN: echo '{' > %t.json
// RUN: echo '  "married" : true' >> %t.json
// RUN: echo '}' >> %t.json

// RUN: clang-format -n -style=LLVM < %t.json 2>&1 | FileCheck %s -allow-empty

// RUN: clang-format -n -style=LLVM %t.json 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK2 -strict-whitespace

// RUN: rm %t.json

// CHECK-NOT: warning
// CHECK2: warning: code should be clang-formatted
