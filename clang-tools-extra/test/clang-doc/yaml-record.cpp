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

// RUN: clang-doc --format=yaml --doxygen -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./C.yaml | FileCheck %s --check-prefix CHECK-0
// CHECK-0: ---
// CHECK-0-NEXT: USR:             '06B5F6A19BA9F6A832E127C9968282B94619B210'
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
// CHECK-1-NEXT: USR:             'ACE81AFA6627B4CEF2B456FB6E1252925674AF7E'
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
// CHECK-2-NEXT: USR:             'E3B54702FABFF4037025BA194FC27C47006330B5'
// CHECK-2-NEXT: Name:            'F'
// CHECK-2-NEXT: DefLocation:     
// CHECK-2-NEXT:   LineNumber:      36
// CHECK-2-NEXT:   Filename:        'test'
// CHECK-2-NEXT: TagType:         Class
// CHECK-2-NEXT: Parents:         
// CHECK-2-NEXT:   - Type:            Record
// CHECK-2-NEXT:     Name:            'E'
// CHECK-2-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-2-NEXT: VirtualParents:  
// CHECK-2-NEXT:   - Type:            Record
// CHECK-2-NEXT:     Name:            'D'
// CHECK-2-NEXT:     USR:             '0921737541208B8FA9BB42B60F78AC1D779AA054'
// CHECK-2-NEXT: ...

// RUN: cat %t/docs/./E.yaml | FileCheck %s --check-prefix CHECK-3
// CHECK-3: ---
// CHECK-3-NEXT: USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-3-NEXT: Name:            'E'
// CHECK-3-NEXT: DefLocation:     
// CHECK-3-NEXT:   LineNumber:      25
// CHECK-3-NEXT:   Filename:        'test'
// CHECK-3-NEXT: TagType:         Class
// CHECK-3-NEXT: ...

// RUN: cat %t/docs/./D.yaml | FileCheck %s --check-prefix CHECK-4
// CHECK-4: ---
// CHECK-4-NEXT: USR:             '0921737541208B8FA9BB42B60F78AC1D779AA054'
// CHECK-4-NEXT: Name:            'D'
// CHECK-4-NEXT: DefLocation:     
// CHECK-4-NEXT:   LineNumber:      23
// CHECK-4-NEXT:   Filename:        'test'
// CHECK-4-NEXT: TagType:         Class
// CHECK-4-NEXT: ...

// RUN: cat %t/docs/./B.yaml | FileCheck %s --check-prefix CHECK-5
// CHECK-5: ---
// CHECK-5-NEXT: USR:             'FC07BD34D5E77782C263FA944447929EA8753740'
// CHECK-5-NEXT: Name:            'B'
// CHECK-5-NEXT: DefLocation:     
// CHECK-5-NEXT:   LineNumber:      17
// CHECK-5-NEXT:   Filename:        'test'
// CHECK-5-NEXT: Members:         
// CHECK-5-NEXT:   - 'X'
// CHECK-5-NEXT:   - 'Y'
// CHECK-5-NEXT: ...

// RUN: cat %t/docs/./X.yaml | FileCheck %s --check-prefix CHECK-6
// CHECK-6: ---
// CHECK-6-NEXT: USR:             'CA7C7935730B5EACD25F080E9C83FA087CCDC75E'
// CHECK-6-NEXT: Name:            'X'
// CHECK-6-NEXT: DefLocation:     
// CHECK-6-NEXT:   LineNumber:      38
// CHECK-6-NEXT:   Filename:        'test'
// CHECK-6-NEXT: TagType:         Class
// CHECK-6-NEXT: ...

// RUN: cat %t/docs/./H.yaml | FileCheck %s --check-prefix CHECK-7
// CHECK-7: ---
// CHECK-7-NEXT: USR:             'B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E'
// CHECK-7-NEXT: Name:            'H'
// CHECK-7-NEXT: DefLocation:     
// CHECK-7-NEXT:   LineNumber:      11
// CHECK-7-NEXT:   Filename:        'test'
// CHECK-7-NEXT: ReturnType:      
// CHECK-7-NEXT:   Type:            
// CHECK-7-NEXT:     Name:            'void'
// CHECK-7-NEXT: ...

// RUN: cat %t/docs/./Bc.yaml | FileCheck %s --check-prefix CHECK-8
// CHECK-8: ---
// CHECK-8-NEXT: USR:             '1E3438A08BA22025C0B46289FF0686F92C8924C5'
// CHECK-8-NEXT: Name:            'Bc'
// CHECK-8-NEXT: DefLocation:     
// CHECK-8-NEXT:   LineNumber:      19
// CHECK-8-NEXT:   Filename:        'test'
// CHECK-8-NEXT: Scoped:          true
// CHECK-8-NEXT: Members:         
// CHECK-8-NEXT:   - 'A'
// CHECK-8-NEXT:   - 'B'
// CHECK-8-NEXT: ...

// RUN: cat %t/docs/H/I.yaml | FileCheck %s --check-prefix CHECK-9
// CHECK-9: ---
// CHECK-9-NEXT: USR:             '3FB542274573CAEAD54CEBFFCAEE3D77FB9713D8'
// CHECK-9-NEXT: Name:            'I'
// CHECK-9-NEXT: Namespace:       
// CHECK-9-NEXT:   - Type:            Function
// CHECK-9-NEXT:     Name:            'H'
// CHECK-9-NEXT:     USR:             'B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E'
// CHECK-9-NEXT: DefLocation:     
// CHECK-9-NEXT:   LineNumber:      12
// CHECK-9-NEXT:   Filename:        'test'
// CHECK-9-NEXT: TagType:         Class
// CHECK-9-NEXT: ...

// RUN: cat %t/docs/X/Y.yaml | FileCheck %s --check-prefix CHECK-10
// CHECK-10: ---
// CHECK-10-NEXT: USR:             '641AB4A3D36399954ACDE29C7A8833032BF40472'
// CHECK-10-NEXT: Name:            'Y'
// CHECK-10-NEXT: Namespace:       
// CHECK-10-NEXT:   - Type:            Record
// CHECK-10-NEXT:     Name:            'X'
// CHECK-10-NEXT:     USR:             'CA7C7935730B5EACD25F080E9C83FA087CCDC75E'
// CHECK-10-NEXT: DefLocation:     
// CHECK-10-NEXT:   LineNumber:      39
// CHECK-10-NEXT:   Filename:        'test'
// CHECK-10-NEXT: TagType:         Class
// CHECK-10-NEXT: ...

// RUN: cat %t/docs/E/ProtectedMethod.yaml | FileCheck %s --check-prefix CHECK-11
// CHECK-11: ---
// CHECK-11-NEXT: USR:             '5093D428CDC62096A67547BA52566E4FB9404EEE'
// CHECK-11-NEXT: Name:            'ProtectedMethod'
// CHECK-11-NEXT: Namespace:       
// CHECK-11-NEXT:   - Type:            Record
// CHECK-11-NEXT:     Name:            'E'
// CHECK-11-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-11-NEXT: DefLocation:     
// CHECK-11-NEXT:   LineNumber:      34
// CHECK-11-NEXT:   Filename:        'test'
// CHECK-11-NEXT: Location:        
// CHECK-11-NEXT:   - LineNumber:      31
// CHECK-11-NEXT:     Filename:        'test'
// CHECK-11-NEXT: IsMethod:        true
// CHECK-11-NEXT: Parent:          
// CHECK-11-NEXT:   Type:            Record
// CHECK-11-NEXT:   Name:            'E'
// CHECK-11-NEXT:   USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-11-NEXT: ReturnType:      
// CHECK-11-NEXT:   Type:            
// CHECK-11-NEXT:     Name:            'void'
// CHECK-11-NEXT: ...

// RUN: cat %t/docs/E/E.yaml | FileCheck %s --check-prefix CHECK-12
// CHECK-12: ---
// CHECK-12-NEXT: USR:             'DEB4AC1CD9253CD9EF7FBE6BCAC506D77984ABD4'
// CHECK-12-NEXT: Name:            'E'
// CHECK-12-NEXT: Namespace:       
// CHECK-12-NEXT:   - Type:            Record
// CHECK-12-NEXT:     Name:            'E'
// CHECK-12-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-12-NEXT: DefLocation:     
// CHECK-12-NEXT:   LineNumber:      27
// CHECK-12-NEXT:   Filename:        'test'
// CHECK-12-NEXT: IsMethod:        true
// CHECK-12-NEXT: Parent:          
// CHECK-12-NEXT:   Type:            Record
// CHECK-12-NEXT:   Name:            'E'
// CHECK-12-NEXT:   USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-12-NEXT: ReturnType:      
// CHECK-12-NEXT:   Type:            
// CHECK-12-NEXT:     Name:            'void'
// CHECK-12-NEXT: ...

// RUN: cat %t/docs/E/~E.yaml | FileCheck %s --check-prefix CHECK-13
// CHECK-13: ---
// CHECK-13-NEXT: USR:             'BD2BDEBD423F80BACCEA75DE6D6622D355FC2D17'
// CHECK-13-NEXT: Name:            '~E'
// CHECK-13-NEXT: Namespace:       
// CHECK-13-NEXT:   - Type:            Record
// CHECK-13-NEXT:     Name:            'E'
// CHECK-13-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-13-NEXT: DefLocation:     
// CHECK-13-NEXT:   LineNumber:      28
// CHECK-13-NEXT:   Filename:        'test'
// CHECK-13-NEXT: IsMethod:        true
// CHECK-13-NEXT: Parent:          
// CHECK-13-NEXT:   Type:            Record
// CHECK-13-NEXT:   Name:            'E'
// CHECK-13-NEXT:   USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-13-NEXT: ReturnType:      
// CHECK-13-NEXT:   Type:            
// CHECK-13-NEXT:     Name:            'void'
// CHECK-13-NEXT: ...
