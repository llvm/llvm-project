// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --doxygen --executor=standalone -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/GlobalNamespace/index.yaml | FileCheck %s --check-prefix=CHECK
// RUN: rm -rf %t

void function(int x);

void function(int x) {}

// CHECK: ---
// CHECK-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-NEXT: ChildFunctions:
// CHECK-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-NEXT:    Name:            'function'
// CHECK-NEXT:    DefLocation:
// CHECK-NEXT:      LineNumber:      [[@LINE-8]]
// CHECK-NEXT:      Filename:        '{{.*}}
// CHECK-NEXT:    Location:
// CHECK-NEXT:      - LineNumber:      [[@LINE-13]]
// CHECK-NEXT:        Filename:        '{{.*}}'
// CHECK-NEXT:    Params:
// CHECK-NEXT:      - Type:
// CHECK-NEXT:          Name:            'int'
// CHECK-NEXT:        Name:            'x'
// CHECK-NEXT:    ReturnType:
// CHECK-NEXT:      Type:
// CHECK-NEXT:        Name:            'void'
// CHECK-NEXT:...
