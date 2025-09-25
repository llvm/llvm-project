// RUN: rm -rf %t
// RUN: mkdir -p %t/yaml %t/md

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/yaml
// RUN: FileCheck %s < %t/yaml/index.yaml --check-prefix=YAML

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/md --format=md
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md  --check-prefix=MD

//      YAML: ---
// YAML-NEXT: USR:             '0000000000000000000000000000000000000000'
// YAML-NEXT: ChildFunctions:

// MD: # Global Namespace
// MD: ## Functions

extern bool b();

// YAML-NEXT:   - USR:             '88A104C263241E354ECF5B55B04AE8CEAD625B71'
// YAML-NEXT:     Name:            'b'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'bool'
// YAML-NEXT:         QualName:        'bool'

// MD: ### b
// MD: *bool b()*

char c();

// YAML-NEXT:   - USR:             'EA3287837B3F175C8DB154406B4DAD2924F479B5'
// YAML-NEXT:     Name:            'c'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'char'
// YAML-NEXT:         QualName:        'char'

// MD: ### c
// MD: *char c()*

double d();

// YAML-NEXT:   - USR:             '60A47E4696CEFC411AB2E1EEFA2DD914E2A7E450'
// YAML-NEXT:     Name:            'd'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'double'
// YAML-NEXT:         QualName:        'double'

// MD: ### d
// MD: *double d()*

float f();

// YAML-NEXT:   - USR:             'B3A9EC6BECD5869CF3ACDFB25153CFE6BBDD5EAB'
// YAML-NEXT:     Name:            'f'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'float'
// YAML-NEXT:         QualName:        'float'

// MD: ### f
// MD: *float f()*

int i();

// YAML-NEXT:   - USR:             '307041280A81EB46F949A94AD52587C659FD801C'
// YAML-NEXT:     Name:            'i'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'int'
// YAML-NEXT:         QualName:        'int'

// MD: ### i
// MD: *int i()*

long l();

// YAML-NEXT:   - USR:             'A1CE9AB0064C412F857592E01332C641C1A06F37'
// YAML-NEXT:     Name:            'l'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'long'
// YAML-NEXT:         QualName:        'long'

// MD: ### l
// MD: *long l()*

long long ll();

// YAML-NEXT:   - USR:             '5C2C44ED4825C066EF6ED796863586F343C8BCA9'
// YAML-NEXT:     Name:            'll'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'long long'
// YAML-NEXT:         QualName:        'long long'

// MD: ### ll
// MD: *long long ll()*

short s();

// YAML-NEXT:   - USR:             '412341570FD3AD2C3A1E9A1DE7B3C01C07BEACFE'
// YAML-NEXT:     Name:            's'
// YAML-NEXT:     Location:
// YAML-NEXT:       - LineNumber:      [[# @LINE-5]]
// YAML-NEXT:         Filename:        '{{.*}}'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'short'
// YAML-NEXT:         QualName:        'short'
// YAML-NEXT: ...

// MD: ### s
// MD: *short s()*
