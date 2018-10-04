// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// This test requires Linux due to system-dependent USR for the inner class.
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

void H() {
  class I {};
}

union A { int X; int Y; };

enum B { X, Y };

enum class Bc { A, B };

struct C { int i; };

class D {};

class E {
public:
  E() {}
  ~E() {}

protected:
  void ProtectedMethod();
};

void E::ProtectedMethod() {}

class F : virtual private D, public E {};

class X {
  class Y {};
};

class G;

// RUN: clang-doc --format=yaml --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./C.yaml | FileCheck %s --check-prefix CHECK-0
// CHECK-0: ---
// CHECK-0-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT: Name:            'C'
// CHECK-0-NEXT: DefLocation:     
// CHECK-0-NEXT:   LineNumber:      21
// CHECK-0-NEXT:   Filename:        'test'
// CHECK-0-NEXT: Members:         
// CHECK-0-NEXT:   - Type:            
// CHECK-0-NEXT:       Name:            'int'
// CHECK-0-NEXT:     Name:            'i'
// CHECK-0-NEXT: ...

// RUN: cat %t/docs/./A.yaml | FileCheck %s --check-prefix CHECK-1
// CHECK-1: ---
// CHECK-1-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-1-NEXT: Name:            'A'
// CHECK-1-NEXT: DefLocation:     
// CHECK-1-NEXT:   LineNumber:      15
// CHECK-1-NEXT:   Filename:        'test'
// CHECK-1-NEXT: TagType:         Union
// CHECK-1-NEXT: Members:         
// CHECK-1-NEXT:   - Type:            
// CHECK-1-NEXT:       Name:            'int'
// CHECK-1-NEXT:     Name:            'X'
// CHECK-1-NEXT:   - Type:            
// CHECK-1-NEXT:       Name:            'int'
// CHECK-1-NEXT:     Name:            'Y'
// CHECK-1-NEXT: ...

// RUN: cat %t/docs/./F.yaml | FileCheck %s --check-prefix CHECK-2
// CHECK-2: ---
// CHECK-2-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT: Name:            'F'
// CHECK-2-NEXT: DefLocation:     
// CHECK-2-NEXT:   LineNumber:      36
// CHECK-2-NEXT:   Filename:        'test'
// CHECK-2-NEXT: TagType:         Class
// CHECK-2-NEXT: Parents:         
// CHECK-2-NEXT:   - Type:            Record
// CHECK-2-NEXT:     Name:            'E'
// CHECK-2-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT: VirtualParents:  
// CHECK-2-NEXT:   - Type:            Record
// CHECK-2-NEXT:     Name:            'D'
// CHECK-2-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT: ...

// RUN: cat %t/docs/./G.yaml | FileCheck %s --check-prefix CHECK-3
// CHECK-3: ---
// CHECK-3-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT: Name:            'G'
// CHECK-3-NEXT: Location:        
// CHECK-3-NEXT:   - LineNumber:      42
// CHECK-3-NEXT:     Filename:        'test'
// CHECK-3-NEXT: TagType:         Class
// CHECK-3-NEXT: ...

// RUN: cat %t/docs/./E.yaml | FileCheck %s --check-prefix CHECK-4
// CHECK-4: ---
// CHECK-4-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT: Name:            'E'
// CHECK-4-NEXT: DefLocation:     
// CHECK-4-NEXT:   LineNumber:      25
// CHECK-4-NEXT:   Filename:        'test'
// CHECK-4-NEXT: TagType:         Class
// CHECK-4-NEXT: ChildFunctions:  
// CHECK-4-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     Name:            'E'
// CHECK-4-NEXT:     Namespace:       
// CHECK-4-NEXT:       - Type:            Record
// CHECK-4-NEXT:         Name:            'E'
// CHECK-4-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     DefLocation:     
// CHECK-4-NEXT:       LineNumber:      27
// CHECK-4-NEXT:       Filename:        'test'
// CHECK-4-NEXT:     IsMethod:        true
// CHECK-4-NEXT:     Parent:          
// CHECK-4-NEXT:       Type:            Record
// CHECK-4-NEXT:       Name:            'E'
// CHECK-4-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     ReturnType:      
// CHECK-4-NEXT:       Type:            
// CHECK-4-NEXT:         Name:            'void'
// CHECK-4-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     Name:            '~E'
// CHECK-4-NEXT:     Namespace:       
// CHECK-4-NEXT:       - Type:            Record
// CHECK-4-NEXT:         Name:            'E'
// CHECK-4-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     DefLocation:     
// CHECK-4-NEXT:       LineNumber:      28
// CHECK-4-NEXT:       Filename:        'test'
// CHECK-4-NEXT:     IsMethod:        true
// CHECK-4-NEXT:     Parent:          
// CHECK-4-NEXT:       Type:            Record
// CHECK-4-NEXT:       Name:            'E'
// CHECK-4-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     ReturnType:      
// CHECK-4-NEXT:       Type:            
// CHECK-4-NEXT:         Name:            'void'
// CHECK-4-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     Name:            'ProtectedMethod'
// CHECK-4-NEXT:     Namespace:       
// CHECK-4-NEXT:       - Type:            Record
// CHECK-4-NEXT:         Name:            'E'
// CHECK-4-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     DefLocation:     
// CHECK-4-NEXT:       LineNumber:      34
// CHECK-4-NEXT:       Filename:        'test'
// CHECK-4-NEXT:     Location:        
// CHECK-4-NEXT:       - LineNumber:      31
// CHECK-4-NEXT:         Filename:        'test'
// CHECK-4-NEXT:     IsMethod:        true
// CHECK-4-NEXT:     Parent:          
// CHECK-4-NEXT:       Type:            Record
// CHECK-4-NEXT:       Name:            'E'
// CHECK-4-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-4-NEXT:     ReturnType:      
// CHECK-4-NEXT:       Type:            
// CHECK-4-NEXT:         Name:            'void'
// CHECK-4-NEXT: ...

// RUN: cat %t/docs/./D.yaml | FileCheck %s --check-prefix CHECK-5
// CHECK-5: ---
// CHECK-5-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-5-NEXT: Name:            'D'
// CHECK-5-NEXT: DefLocation:     
// CHECK-5-NEXT:   LineNumber:      23
// CHECK-5-NEXT:   Filename:        'test'
// CHECK-5-NEXT: TagType:         Class
// CHECK-5-NEXT: ...

// RUN: cat %t/docs/./X.yaml | FileCheck %s --check-prefix CHECK-6
// CHECK-6: ---
// CHECK-6-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-6-NEXT: Name:            'X'
// CHECK-6-NEXT: DefLocation:     
// CHECK-6-NEXT:   LineNumber:      38
// CHECK-6-NEXT:   Filename:        'test'
// CHECK-6-NEXT: TagType:         Class
// CHECK-6-NEXT: ...

// RUN: cat %t/docs/./GlobalNamespace.yaml | FileCheck %s --check-prefix CHECK-7
// CHECK-7: ---
// CHECK-7-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-7-NEXT: ChildFunctions:  
// CHECK-7-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-7-NEXT:     Name:            'H'
// CHECK-7-NEXT:     DefLocation:     
// CHECK-7-NEXT:       LineNumber:      11
// CHECK-7-NEXT:       Filename:        'test'
// CHECK-7-NEXT:     ReturnType:      
// CHECK-7-NEXT:       Type:            
// CHECK-7-NEXT:         Name:            'void'
// CHECK-7-NEXT: ChildEnums:      
// CHECK-7-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-7-NEXT:     Name:            'B'
// CHECK-7-NEXT:     DefLocation:     
// CHECK-7-NEXT:       LineNumber:      17
// CHECK-7-NEXT:       Filename:        'test'
// CHECK-7-NEXT:     Members:         
// CHECK-7-NEXT:       - 'X'
// CHECK-7-NEXT:       - 'Y'
// CHECK-7-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-7-NEXT:     Name:            'Bc'
// CHECK-7-NEXT:     DefLocation:     
// CHECK-7-NEXT:       LineNumber:      19
// CHECK-7-NEXT:       Filename:        'test'
// CHECK-7-NEXT:     Scoped:          true
// CHECK-7-NEXT:     Members:         
// CHECK-7-NEXT:       - 'A'
// CHECK-7-NEXT:       - 'B'
// CHECK-7-NEXT: ...

// RUN: cat %t/docs/X/Y.yaml | FileCheck %s --check-prefix CHECK-8
// CHECK-8: ---
// CHECK-8-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-8-NEXT: Name:            'Y'
// CHECK-8-NEXT: Namespace:       
// CHECK-8-NEXT:   - Type:            Record
// CHECK-8-NEXT:     Name:            'X'
// CHECK-8-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-8-NEXT: DefLocation:     
// CHECK-8-NEXT:   LineNumber:      39
// CHECK-8-NEXT:   Filename:        'test'
// CHECK-8-NEXT: TagType:         Class
// CHECK-8-NEXT: ...
