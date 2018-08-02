// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

namespace A {
  
void f();

}  // namespace A

namespace A {

void f(){};

namespace B {

enum E { X };

E func(int i) { return X; }

}  // namespace B
}  // namespace A

// RUN: clang-doc --format=yaml --doxygen -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./A.yaml | FileCheck %s --check-prefix CHECK-0
// CHECK-0: ---
// CHECK-0-NEXT: USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-0-NEXT: Name:            'A'
// CHECK-0-NEXT: ...

// RUN: cat %t/docs/A/f.yaml | FileCheck %s --check-prefix CHECK-1
// CHECK-1: ---
// CHECK-1-NEXT: USR:             '39D3C95A5F7CE2BA4937BD7B01BAE09EBC2AD8AC'
// CHECK-1-NEXT: Name:            'f'
// CHECK-1-NEXT: Namespace:       
// CHECK-1-NEXT:   - Type:            Namespace
// CHECK-1-NEXT:     Name:            'A'
// CHECK-1-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-1-NEXT: DefLocation:     
// CHECK-1-NEXT:   LineNumber:      17
// CHECK-1-NEXT:   Filename:        'test'
// CHECK-1-NEXT: Location:        
// CHECK-1-NEXT:   - LineNumber:      11
// CHECK-1-NEXT:     Filename:        'test'
// CHECK-1-NEXT: ReturnType:      
// CHECK-1-NEXT:   Type:            
// CHECK-1-NEXT:     Name:            'void'
// CHECK-1-NEXT: ...

// RUN: cat %t/docs/A/B.yaml | FileCheck %s --check-prefix CHECK-2
// CHECK-2: ---
// CHECK-2-NEXT: USR:             'E21AF79E2A9D02554BA090D10DF39FE273F5CDB5'
// CHECK-2-NEXT: Name:            'B'
// CHECK-2-NEXT: Namespace:       
// CHECK-2-NEXT:   - Type:            Namespace
// CHECK-2-NEXT:     Name:            'A'
// CHECK-2-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-2-NEXT: ...

// RUN: cat %t/docs/A/B/E.yaml | FileCheck %s --check-prefix CHECK-3
// CHECK-3: ---
// CHECK-3-NEXT: USR:             'E9ABF7E7E2425B626723D41E76E4BC7E7A5BD775'
// CHECK-3-NEXT: Name:            'E'
// CHECK-3-NEXT: Namespace:       
// CHECK-3-NEXT:   - Type:            Namespace
// CHECK-3-NEXT:     Name:            'B'
// CHECK-3-NEXT:     USR:             'E21AF79E2A9D02554BA090D10DF39FE273F5CDB5'
// CHECK-3-NEXT:   - Type:            Namespace
// CHECK-3-NEXT:     Name:            'A'
// CHECK-3-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-3-NEXT: DefLocation:     
// CHECK-3-NEXT:   LineNumber:      21
// CHECK-3-NEXT:   Filename:        'test'
// CHECK-3-NEXT: Members:         
// CHECK-3-NEXT:   - 'X'
// CHECK-3-NEXT: ...

// RUN: cat %t/docs/A/B/func.yaml | FileCheck %s --check-prefix CHECK-4
// CHECK-4: ---
// CHECK-4-NEXT: USR:             '9A82CB33ED0FDF81EE383D31CD0957D153C5E840'
// CHECK-4-NEXT: Name:            'func'
// CHECK-4-NEXT: Namespace:       
// CHECK-4-NEXT:   - Type:            Namespace
// CHECK-4-NEXT:     Name:            'B'
// CHECK-4-NEXT:     USR:             'E21AF79E2A9D02554BA090D10DF39FE273F5CDB5'
// CHECK-4-NEXT:   - Type:            Namespace
// CHECK-4-NEXT:     Name:            'A'
// CHECK-4-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-4-NEXT: DefLocation:     
// CHECK-4-NEXT:   LineNumber:      23
// CHECK-4-NEXT:   Filename:        'test'
// CHECK-4-NEXT: Params:          
// CHECK-4-NEXT:   - Type:            
// CHECK-4-NEXT:       Name:            'int'
// CHECK-4-NEXT:     Name:            'i'
// CHECK-4-NEXT: ReturnType:      
// CHECK-4-NEXT:   Type:            
// CHECK-4-NEXT:     Name:            'enum A::B::E'
// CHECK-4-NEXT: ...
