// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --doxygen --executor=standalone -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/index.yaml | FileCheck %s --check-prefix=CHECK
// RUN: rm -rf %t

template<typename T, int U = 1>
void function(T x) {}

template<>
void function<bool, 0>(bool x) {}

template<class... T>
void ParamPackFunction(T... args);

// CHECK: ---
// CHECK-NEXT: USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT: ChildFunctions:
// CHECK-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT:     Name:            'function'
// CHECK-NEXT:     DefLocation:
// CHECK-NEXT:       LineNumber:      10
// CHECK-NEXT:       Filename:        '{{.*}}'
// CHECK-NEXT:     Params:
// CHECK-NEXT:       - Type:
// CHECK-NEXT:           Name:            'T'
// CHECK-NEXT:           QualName:        'T'
// CHECK-NEXT:         Name:            'x'
// CHECK-NEXT:     ReturnType:
// CHECK-NEXT:       Type:
// CHECK-NEXT:         Name:            'void'
// CHECK-NEXT:         QualName:        'void'
// CHECK-NEXT:     Template:
// CHECK-NEXT:       Params:
// CHECK-NEXT:         - Contents:        'typename T'
// CHECK-NEXT:         - Contents:        'int U = 1'
// CHECK-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT:     Name:            'function'
// CHECK-NEXT:     DefLocation:
// CHECK-NEXT:       LineNumber:      12
// CHECK-NEXT:       Filename:        '{{.*}}'
// CHECK-NEXT:     Params:
// CHECK-NEXT:       - Type:
// CHECK-NEXT:           Name:            '_Bool'
// CHECK-NEXT:           QualName:        '_Bool'
// CHECK-NEXT:         Name:            'x'
// CHECK-NEXT:     ReturnType:
// CHECK-NEXT:       Type:
// CHECK-NEXT:         Name:            'void'
// CHECK-NEXT:         QualName:        'void'
// CHECK-NEXT:     Template:
// CHECK-NEXT:       Specialization:
// CHECK-NEXT:         SpecializationOf: '{{([0-9A-F]{40})}}'
// CHECK-NEXT:         Params:
// CHECK-NEXT:           - Contents:        'bool'
// CHECK-NEXT:           - Contents:        '0'
// CHECK-NEXT:  - USR:             '{{([0-9A-F]{40})}}'
// CHECK-NEXT:    Name:            'ParamPackFunction'
// CHECK-NEXT:    Location:
// CHECK-NEXT:      - LineNumber:      16
// CHECK-NEXT:        Filename:        '{{.*}}'
// CHECK-NEXT:    Params:
// CHECK-NEXT:      - Type:
// CHECK-NEXT:          Name:            'T...'
// CHECK-NEXT:          QualName:        'T...'
// CHECK-NEXT:        Name:            'args'
// CHECK-NEXT:    ReturnType:
// CHECK-NEXT:      Type:
// CHECK-NEXT:        Name:            'void'
// CHECK-NEXT:        QualName:        'void'
// CHECK-NEXT:    Template:
// CHECK-NEXT:      Params:
// CHECK-NEXT:        - Contents:        'class... T'
// CHECK-NEXT: ...
