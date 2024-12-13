// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --doxygen --executor=standalone -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/index.yaml | FileCheck %s --check-prefix=YAML
// RUN: rm -rf %t

template<typename T, int U = 1>
void function(T x) {}

template<>
void function<bool, 0>(bool x) {}

template<class... T>
void ParamPackFunction(T... args);

// YAML: ---
// YAML-NEXT: USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT: ChildFunctions:
// YAML-NEXT:  - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'ParamPackFunction'
// YAML-NEXT:    Location:
// YAML-NEXT:      - LineNumber:      16
// YAML-NEXT:        Filename:        '{{.*}}'
// YAML-NEXT:    Params:
// YAML-NEXT:      - Type:
// YAML-NEXT:          Name:            'T...'
// YAML-NEXT:          QualName:        'T...'
// YAML-NEXT:        Name:            'args'
// YAML-NEXT:    ReturnType:
// YAML-NEXT:      Type:
// YAML-NEXT:        Name:            'void'
// YAML-NEXT:        QualName:        'void'
// YAML-NEXT:    Template:
// YAML-NEXT:      Params:
// YAML-NEXT:        - Contents:        'class... T'
// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      10
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'T'
// YAML-NEXT:           QualName:        'T'
// YAML-NEXT:         Name:            'x'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'void'
// YAML-NEXT:         QualName:        'void'
// YAML-NEXT:     Template:
// YAML-NEXT:       Params:
// YAML-NEXT:         - Contents:        'typename T'
// YAML-NEXT:         - Contents:        'int U = 1'
// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      12
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            '_Bool'
// YAML-NEXT:           QualName:        '_Bool'
// YAML-NEXT:         Name:            'x'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'void'
// YAML-NEXT:         QualName:        'void'
// YAML-NEXT:     Template:
// YAML-NEXT:       Specialization:
// YAML-NEXT:         SpecializationOf: '{{([0-9A-F]{40})}}'
// YAML-NEXT:         Params:
// YAML-NEXT:           - Contents:        'bool'
// YAML-NEXT:           - Contents:        '0'
// YAML-NEXT: ...
