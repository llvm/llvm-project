// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --doxygen --executor=standalone %S/../Inputs/templates.cpp -output=%t/docs
// RUN: cat %t/docs/index.yaml | FileCheck %s --check-prefix=YAML

// YAML: ---
// YAML-NEXT: USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT: ChildRecords:
// YAML-NEXT:   - Type:            Record
// YAML-NEXT:     Name:            'tuple'
// YAML-NEXT:     QualName:        'tuple'
// YAML-NEXT:     USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Path:            'GlobalNamespace'
// YAML-NEXT: ChildFunctions:
// YAML-NEXT:  - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'ParamPackFunction'
// YAML-NEXT:    Location:
// YAML-NEXT:      - LineNumber:      1
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
// YAML-NEXT:       LineNumber:      3
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
// YAML-NEXT:     Name:            'longFunction'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      6
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'A'
// YAML-NEXT:           QualName:        'A'
// YAML-NEXT:         Name:            'a'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'B'
// YAML-NEXT:           QualName:        'B'
// YAML-NEXT:         Name:            'b'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'C'
// YAML-NEXT:           QualName:        'C'
// YAML-NEXT:         Name:            'c'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'D'
// YAML-NEXT:           QualName:        'D'
// YAML-NEXT:         Name:            'd'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'E'
// YAML-NEXT:           QualName:        'E'
// YAML-NEXT:         Name:            'e'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'void'
// YAML-NEXT:         QualName:        'void'
// YAML-NEXT:     Template:
// YAML-NEXT:         Params:
// YAML-NEXT:           - Contents:        'typename A'
// YAML-NEXT:           - Contents:        'typename B'
// YAML-NEXT:           - Contents:        'typename C'
// YAML-NEXT:           - Contents:        'typename D'
// YAML-NEXT:           - Contents:        'typename E'

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      8
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'bool'
// YAML-NEXT:           QualName:        'bool'
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

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'func_with_tuple_param'
// YAML-NEXT:    Description:
// YAML-NEXT:      - Kind:            FullComment
// YAML-NEXT:        Children:
// YAML-NEXT:          - Kind:            ParagraphComment
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            TextComment
// YAML-NEXT:                Text:            'A function with a tuple parameter'
// YAML-NEXT:          - Kind:            ParagraphComment
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            TextComment
// YAML-NEXT:          - Kind:            ParamCommandComment
// YAML-NEXT:            Direction:       '[in]'
// YAML-NEXT:            ParamName:       't'
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            ParagraphComment
// YAML-NEXT:                Children:
// YAML-NEXT:                  - Kind:            TextComment
// YAML-NEXT:                    Text:            'The input to func_with_tuple_param'
// YAML-NEXT:    DefLocation:
// YAML-NEXT:      LineNumber:      18
// YAML-NEXT:      Filename:
// YAML-NEXT:    Params:
// YAML-NEXT:      - Type:
// YAML-NEXT:          Type:            Record
// YAML-NEXT:          Name:            'tuple'
// YAML-NEXT:          QualName:        'tuple<int, int, bool>'
// YAML-NEXT:          USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:          Path:            'GlobalNamespace'
// YAML-NEXT:        Name:            't'
// YAML-NEXT:    ReturnType:
// YAML-NEXT:      Type:
// YAML-NEXT:        Type:            Record
// YAML-NEXT:        Name:            'tuple'
// YAML-NEXT:        QualName:        'tuple<int, int, bool>'
// YAML-NEXT:        USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:        Path:            'GlobalNamespace'
// YAML-NEXT: ...
