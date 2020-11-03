// RUN: clang-tidy -dump-config | FileCheck %s
// RUN: clang-tidy -dump-config -use-color | FileCheck -check-prefix=CHECK-CONFIG-COLOR %s
// RUN: clang-tidy -dump-config -use-color=false | FileCheck -check-prefix=CHECK-CONFIG-NO-COLOR %s
// RUN: clang-tidy -config='UseColor: true' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-COLOR %s
// RUN: clang-tidy -config='UseColor: false' -dump-config | FileCheck -check-prefix=CHECK-CONFIG-NO-COLOR %s
// RUN: clang-tidy -help | FileCheck -check-prefix=CHECK-OPT-PRESENT %s

// RUN: clang-tidy -checks='-*, modernize-use-override' -extra-arg=-std=c++11 -use-color=false %s | FileCheck -check-prefix=CHECK-NO-COLOR %s
// RUN: clang-tidy -checks='-*, modernize-use-override' -extra-arg=-std=c++11 %s | FileCheck -check-prefix=CHECK-NO-COLOR %s
// RUN: clang-tidy -checks='-*, modernize-use-override' -extra-arg=-std=c++11 -use-color %s | FileCheck -check-prefix=CHECK-COLOR %s

// CHECK-NOT: UseColor
// CHECK-CONFIG-NO-COLOR: UseColor: false
// CHECK-CONFIG-COLOR: UseColor: true
// CHECK-OPT-PRESENT: --use-color

class Base {
public:
  virtual ~Base() = 0;
};

class Delivered : public Base {
public:
  virtual ~Delivered() = default;
  // CHECK-NO-COLOR: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
  // CHECK-COLOR: {{.\[0;1;35m}}warning: {{.\[0m}}{{.\[1m}}prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
};
