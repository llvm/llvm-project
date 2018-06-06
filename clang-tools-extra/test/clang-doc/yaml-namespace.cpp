// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/A.yaml | FileCheck %s --check-prefix=CHECK-A
// RUN: cat %t/docs/A/B.yaml | FileCheck %s --check-prefix=CHECK-B
// RUN: cat %t/docs/A/f.yaml | FileCheck %s --check-prefix=CHECK-F
// RUN: cat %t/docs/A/B/E.yaml | FileCheck %s --check-prefix=CHECK-E
// RUN: cat %t/docs/A/B/func.yaml | FileCheck %s --check-prefix=CHECK-FUNC

namespace A {
  
// CHECK-A: ---
// CHECK-A-NEXT: USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-A-NEXT: Name:            'A'
// CHECK-A-NEXT: ...


void f();

}  // namespace A

namespace A {

void f(){};

// CHECK-F: ---
// CHECK-F-NEXT: USR:             '39D3C95A5F7CE2BA4937BD7B01BAE09EBC2AD8AC'
// CHECK-F-NEXT: Name:            'f'
// CHECK-F-NEXT: Namespace:       
// CHECK-F-NEXT:   - Type:            Namespace
// CHECK-F-NEXT:     Name:            'A'
// CHECK-F-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-F-NEXT: DefLocation:     
// CHECK-F-NEXT:   LineNumber:      26
// CHECK-F-NEXT:   Filename:        '{{.*}}'
// CHECK-F-NEXT: Location:        
// CHECK-F-NEXT:   - LineNumber:      20
// CHECK-F-NEXT:     Filename:        'test'
// CHECK-F-NEXT: ReturnType:      
// CHECK-F-NEXT:   Type:            
// CHECK-F-NEXT:     Name:            'void'
// CHECK-F-NEXT: ...

namespace B {
  
// CHECK-B: ---
// CHECK-B-NEXT: USR:             'E21AF79E2A9D02554BA090D10DF39FE273F5CDB5'
// CHECK-B-NEXT: Name:            'B'
// CHECK-B-NEXT: Namespace:       
// CHECK-B-NEXT:   - Type:            Namespace
// CHECK-B-NEXT:     Name:            'A'
// CHECK-B-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-B-NEXT: ...


enum E { X };

// CHECK-E: ---
// CHECK-E-NEXT: USR:             'E9ABF7E7E2425B626723D41E76E4BC7E7A5BD775'
// CHECK-E-NEXT: Name:            'E'
// CHECK-E-NEXT: Namespace:       
// CHECK-E-NEXT:   - Type:            Namespace
// CHECK-E-NEXT:     Name:            'B'
// CHECK-E-NEXT:     USR:             'E21AF79E2A9D02554BA090D10DF39FE273F5CDB5'
// CHECK-E-NEXT:   - Type:            Namespace
// CHECK-E-NEXT:     Name:            'A'
// CHECK-E-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-E-NEXT: DefLocation:     
// CHECK-E-NEXT:   LineNumber:      58
// CHECK-E-NEXT:   Filename:        '{{.*}}'
// CHECK-E-NEXT: Members:         
// CHECK-E-NEXT:   - 'X'
// CHECK-E-NEXT: ...

E func(int i) { return X; }

// CHECK-FUNC: ---
// CHECK-FUNC-NEXT: USR:             '9A82CB33ED0FDF81EE383D31CD0957D153C5E840'
// CHECK-FUNC-NEXT: Name:            'func'
// CHECK-FUNC-NEXT: Namespace:       
// CHECK-FUNC-NEXT:   - Type:            Namespace
// CHECK-FUNC-NEXT:     Name:            'B'
// CHECK-FUNC-NEXT:     USR:             'E21AF79E2A9D02554BA090D10DF39FE273F5CDB5'
// CHECK-FUNC-NEXT:   - Type:            Namespace
// CHECK-FUNC-NEXT:     Name:            'A'
// CHECK-FUNC-NEXT:     USR:             '8D042EFFC98B373450BC6B5B90A330C25A150E9C'
// CHECK-FUNC-NEXT: DefLocation:     
// CHECK-FUNC-NEXT:   LineNumber:      77
// CHECK-FUNC-NEXT:   Filename:        '{{.*}}'
// CHECK-FUNC-NEXT: Params:          
// CHECK-FUNC-NEXT:   - Type:            
// CHECK-FUNC-NEXT:       Name:            'int'
// CHECK-FUNC-NEXT:     Name:            'i'
// CHECK-FUNC-NEXT: ReturnType:      
// CHECK-FUNC-NEXT:   Type:            
// CHECK-FUNC-NEXT:     Name:            'enum A::B::E'
// CHECK-FUNC-NEXT: ...

}  // namespace B
}  // namespace A
