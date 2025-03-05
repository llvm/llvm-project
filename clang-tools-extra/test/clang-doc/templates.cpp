// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/docs
// RUN: cat %t/docs/index.yaml | FileCheck %s --check-prefix=YAML

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/docs --format=md
// RUN: cat %t/docs/GlobalNamespace/index.md | FileCheck %s --check-prefix=MD

// YAML: ---
// YAML-NEXT: USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT: ChildRecords:
// YAML-NEXT:   - Type:            Record
// YAML-NEXT:     Name:            'tuple'
// YAML-NEXT:     QualName:        'tuple'
// YAML-NEXT:     USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Path:            'GlobalNamespace'

// MD: # Global Namespace
// MD: ## Functions

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

// MD: ### ParamPackFunction
// MD: *void ParamPackFunction(T... args)*

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

// MD: ### function
// MD: *void function(T x)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 23]]*

template<>
void function<bool, 0>(bool x) {}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      [[# @LINE - 6]]
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

// MD: ### function
// MD: *void function(bool x)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 26]]*

/// A Tuple type
///
/// Does Tuple things.
template<typename ...Tys>
struct tuple{};

/// A function with a tuple parameter
///
/// \param t The input to func_with_tuple_param
tuple<int,int,bool> func_with_tuple_param(tuple<int,int,bool> t){ return t;}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'func_with_tuple_param'
// YAML-NEXT:    Description:
// YAML-NEXT:      - Kind:            'FullComment'
// YAML-NEXT:        Children:
// YAML-NEXT:          - Kind:            'ParagraphComment'
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            'TextComment'
// YAML-NEXT:                Text:            ' A function with a tuple parameter'
// YAML-NEXT:          - Kind:            'ParagraphComment'
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            'TextComment'
// YAML-NEXT:          - Kind:            'ParamCommandComment'
// YAML-NEXT:            Direction:       '[in]'
// YAML-NEXT:            ParamName:       't'
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            'ParagraphComment'
// YAML-NEXT:                Children:
// YAML-NEXT:                  - Kind:            'TextComment'
// YAML-NEXT:                    Text:            ' The input to func_with_tuple_param'
// YAML-NEXT:    DefLocation:
// YAML-NEXT:      LineNumber:      [[# @LINE - 23]]
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

// MD: ### func_with_tuple_param
// MD: *tuple<int, int, bool> func_with_tuple_param(tuple<int, int, bool> t)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 44]]*
// MD:  A function with a tuple parameter
// MD: **t** The input to func_with_tuple_param

