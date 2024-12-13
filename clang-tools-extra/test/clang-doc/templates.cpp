// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/docs
// RUN: cat %t/docs/index.yaml | FileCheck %s --check-prefix=YAML

// YAML: ---
// YAML-NEXT: USR:             '{{([0-9A-F]{40})}}'

template<class... T>
void ParamPackFunction(T... args);

// YAML-NEXT: ChildFunctions:
// YAML-NEXT:  - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'ParamPackFunction'
// YAML-NEXT:    Location:
// YAML-NEXT:      - LineNumber:      [[# @LINE - 6]]
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

template<typename T, int U = 1>
void function(T x) {}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      [[# @LINE - 5]]
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

template<>
void function<bool, 0>(bool x) {}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      [[# @LINE - 6]]
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

