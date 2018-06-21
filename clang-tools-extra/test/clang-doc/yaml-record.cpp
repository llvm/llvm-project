// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc -doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: cat %t/docs/A.yaml | FileCheck %s --check-prefix=CHECK-A
// RUN: cat %t/docs/Bc.yaml | FileCheck %s --check-prefix=CHECK-BC
// RUN: cat %t/docs/B.yaml | FileCheck %s --check-prefix=CHECK-B
// RUN: cat %t/docs/C.yaml | FileCheck %s --check-prefix=CHECK-C
// RUN: cat %t/docs/D.yaml | FileCheck %s --check-prefix=CHECK-D
// RUN: cat %t/docs/E.yaml | FileCheck %s --check-prefix=CHECK-E
// RUN: cat %t/docs/E/ProtectedMethod.yaml | FileCheck %s --check-prefix=CHECK-EPM
// RUN: cat %t/docs/E/E.yaml | FileCheck %s --check-prefix=CHECK-ECON
// RUN: cat %t/docs/E/'~E.yaml' | FileCheck %s --check-prefix=CHECK-EDES
// RUN: cat %t/docs/F.yaml | FileCheck %s --check-prefix=CHECK-F
// RUN: cat %t/docs/X.yaml | FileCheck %s --check-prefix=CHECK-X
// RUN: cat %t/docs/X/Y.yaml | FileCheck %s --check-prefix=CHECK-Y
// RUN: cat %t/docs/H.yaml | FileCheck %s --check-prefix=CHECK-H
// RUN: cat %t/docs/H/I.yaml | FileCheck %s --check-prefix=CHECK-I

union A { int X; int Y; };

// CHECK-A: ---
// CHECK-A-NEXT: USR:             'ACE81AFA6627B4CEF2B456FB6E1252925674AF7E'
// CHECK-A-NEXT: Name:            'A'
// CHECK-A-NEXT: DefLocation:     
// CHECK-A-NEXT:   LineNumber:      21
// CHECK-A-NEXT:   Filename:        '{{.*}}'
// CHECK-A-NEXT: TagType:         Union
// CHECK-A-NEXT: Members:         
// CHECK-A-NEXT:   - Type:            
// CHECK-A-NEXT:       Name:            'int'
// CHECK-A-NEXT:     Name:            'X'
// CHECK-A-NEXT:   - Type:            
// CHECK-A-NEXT:       Name:            'int'
// CHECK-A-NEXT:     Name:            'Y'
// CHECK-A-NEXT: ...


enum B { X, Y };

// CHECK-B: ---
// CHECK-B-NEXT: USR:             'FC07BD34D5E77782C263FA944447929EA8753740'
// CHECK-B-NEXT: Name:            'B'
// CHECK-B-NEXT: DefLocation:     
// CHECK-B-NEXT:   LineNumber:      40
// CHECK-B-NEXT:   Filename:        '{{.*}}'
// CHECK-B-NEXT: Members:         
// CHECK-B-NEXT:   - 'X'
// CHECK-B-NEXT:   - 'Y'
// CHECK-B-NEXT: ...

enum class Bc { A, B };

// CHECK-BC: ---
// CHECK-BC-NEXT: USR:             '1E3438A08BA22025C0B46289FF0686F92C8924C5'
// CHECK-BC-NEXT: Name:            'Bc'
// CHECK-BC-NEXT: DefLocation:     
// CHECK-BC-NEXT:   LineNumber:      53
// CHECK-BC-NEXT:   Filename:        '{{.*}}'
// CHECK-BC-NEXT: Scoped:          true
// CHECK-BC-NEXT: Members:         
// CHECK-BC-NEXT:   - 'A'
// CHECK-BC-NEXT:   - 'B'
// CHECK-BC-NEXT: ...

struct C { int i; };

// CHECK-C: ---
// CHECK-C-NEXT: USR:             '06B5F6A19BA9F6A832E127C9968282B94619B210'
// CHECK-C-NEXT: Name:            'C'
// CHECK-C-NEXT: DefLocation:     
// CHECK-C-NEXT:   LineNumber:      67
// CHECK-C-NEXT:   Filename:        '{{.*}}'
// CHECK-C-NEXT: Members:         
// CHECK-C-NEXT:   - Type:            
// CHECK-C-NEXT:       Name:            'int'
// CHECK-C-NEXT:     Name:            'i'
// CHECK-C-NEXT: ...

class D {};

// CHECK-D: ---
// CHECK-D-NEXT: USR:             '0921737541208B8FA9BB42B60F78AC1D779AA054'
// CHECK-D-NEXT: Name:            'D'
// CHECK-D-NEXT: DefLocation:     
// CHECK-D-NEXT:   LineNumber:      81
// CHECK-D-NEXT:   Filename:        '{{.*}}'
// CHECK-D-NEXT: TagType:         Class
// CHECK-D-NEXT: ...

class E {
public:
  E() {}

// CHECK-ECON: ---
// CHECK-ECON-NEXT: USR:             'DEB4AC1CD9253CD9EF7FBE6BCAC506D77984ABD4'
// CHECK-ECON-NEXT: Name:            'E'
// CHECK-ECON-NEXT: Namespace:       
// CHECK-ECON-NEXT:   - Type:            Record
// CHECK-ECON-NEXT:     Name:            'E'
// CHECK-ECON-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-ECON-NEXT: DefLocation:
// CHECK-ECON-NEXT:   LineNumber:      94
// CHECK-ECON-NEXT:   Filename:        '{{.*}}'
// CHECK-ECON-NEXT: IsMethod:        true
// CHECK-ECON-NEXT: Parent:          
// CHECK-ECON-NEXT:   Type:            Record
// CHECK-ECON-NEXT:   Name:            'E'
// CHECK-ECON-NEXT:   USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-ECON-NEXT: ReturnType:      
// CHECK-ECON-NEXT:   Type:            
// CHECK-ECON-NEXT:     Name:            'void'
// CHECK-ECON-NEXT: ...
  
  ~E() {}
  
// CHECK-EDES: ---
// CHECK-EDES-NEXT: USR:             'BD2BDEBD423F80BACCEA75DE6D6622D355FC2D17'
// CHECK-EDES-NEXT: Name:            '~E'
// CHECK-EDES-NEXT: Namespace:       
// CHECK-EDES-NEXT:   - Type:            Record
// CHECK-EDES-NEXT:     Name:            'E'
// CHECK-EDES-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-EDES-NEXT: DefLocation:     
// CHECK-EDES-NEXT:   LineNumber:      116
// CHECK-EDES-NEXT:   Filename:        '{{.*}}'
// CHECK-EDES-NEXT: IsMethod:        true
// CHECK-EDES-NEXT: Parent:          
// CHECK-EDES-NEXT:   Type:            Record
// CHECK-EDES-NEXT:   Name:            'E'
// CHECK-EDES-NEXT:   USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-EDES-NEXT: ReturnType:      
// CHECK-EDES-NEXT:   Type:            
// CHECK-EDES-NEXT:     Name:            'void'
// CHECK-EDES-NEXT: ...


protected:
  void ProtectedMethod();
};

// CHECK-E: ---
// CHECK-E-NEXT: USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-E-NEXT: Name:            'E'
// CHECK-E-NEXT: DefLocation:     
// CHECK-E-NEXT:   LineNumber:      92
// CHECK-E-NEXT:   Filename:        '{{.*}}'
// CHECK-E-NEXT: TagType:         Class
// CHECK-E-NEXT: ...

void E::ProtectedMethod() {}

// CHECK-EPM: ---
// CHECK-EPM-NEXT: USR:             '5093D428CDC62096A67547BA52566E4FB9404EEE'
// CHECK-EPM-NEXT: Name:            'ProtectedMethod'
// CHECK-EPM-NEXT: Namespace:       
// CHECK-EPM-NEXT:   - Type:            Record
// CHECK-EPM-NEXT:     Name:            'E'
// CHECK-EPM-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-EPM-NEXT: DefLocation:     
// CHECK-EPM-NEXT:   LineNumber:      152
// CHECK-EPM-NEXT:   Filename:        '{{.*}}'
// CHECK-EPM-NEXT: Location:        
// CHECK-EPM-NEXT:   - LineNumber:      140
// CHECK-EPM-NEXT:     Filename:        '{{.*}}'
// CHECK-EPM-NEXT: IsMethod:        true
// CHECK-EPM-NEXT: Parent:          
// CHECK-EPM-NEXT:   Type:            Record
// CHECK-EPM-NEXT:   Name:            'E'
// CHECK-EPM-NEXT:   USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-EPM-NEXT: ReturnType:      
// CHECK-EPM-NEXT:   Type:            
// CHECK-EPM-NEXT:     Name:            'void'
// CHECK-EPM-NEXT: ...

class F : virtual private D, public E {};

// CHECK-F: ---
// CHECK-F-NEXT: USR:             'E3B54702FABFF4037025BA194FC27C47006330B5'
// CHECK-F-NEXT: Name:            'F'
// CHECK-F-NEXT: DefLocation:     
// CHECK-F-NEXT:   LineNumber:      177
// CHECK-F-NEXT:   Filename:        '{{.*}}'
// CHECK-F-NEXT: TagType:         Class
// CHECK-F-NEXT: Parents:         
// CHECK-F-NEXT:   - Type:            Record
// CHECK-F-NEXT:     Name:            'E'
// CHECK-F-NEXT:     USR:             '289584A8E0FF4178A794622A547AA622503967A1'
// CHECK-F-NEXT: VirtualParents:  
// CHECK-F-NEXT:   - Type:            Record
// CHECK-F-NEXT:     Name:            'D'
// CHECK-F-NEXT:     USR:             '0921737541208B8FA9BB42B60F78AC1D779AA054'
// CHECK-F-NEXT: ...

class X {
  class Y {};
  
// CHECK-Y: ---
// CHECK-Y-NEXT: USR:             '641AB4A3D36399954ACDE29C7A8833032BF40472'
// CHECK-Y-NEXT: Name:            'Y'
// CHECK-Y-NEXT: Namespace:       
// CHECK-Y-NEXT:   - Type:            Record
// CHECK-Y-NEXT:     Name:            'X'
// CHECK-Y-NEXT:     USR:             'CA7C7935730B5EACD25F080E9C83FA087CCDC75E'
// CHECK-Y-NEXT: DefLocation:     
// CHECK-Y-NEXT:   LineNumber:      197
// CHECK-Y-NEXT:   Filename:        '{{.*}}'
// CHECK-Y-NEXT: TagType:         Class
// CHECK-Y-NEXT: ...

};

// CHECK-X: ---
// CHECK-X-NEXT: USR:             'CA7C7935730B5EACD25F080E9C83FA087CCDC75E'
// CHECK-X-NEXT: Name:            'X'
// CHECK-X-NEXT: DefLocation:     
// CHECK-X-NEXT:   LineNumber:      196
// CHECK-X-NEXT:   Filename:        '{{.*}}'
// CHECK-X-NEXT: TagType:         Class
// CHECK-X-NEXT: ...

void H() {
  class I {};
  
// CHECK-I: ---
// CHECK-I-NEXT: USR:             '{{.*}}'
// CHECK-I-NEXT: Name:            'I'
// CHECK-I-NEXT: Namespace:       
// CHECK-I-NEXT:   - Type:            Function
// CHECK-I-NEXT:     Name:            'H'
// CHECK-I-NEXT:     USR:             'B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E'
// CHECK-I-NEXT: DefLocation:     
// CHECK-I-NEXT:   LineNumber:      224
// CHECK-I-NEXT:   Filename:        'test'
// CHECK-I-NEXT: TagType:         Class
// CHECK-I-NEXT: ...

}

// CHECK-H: ---
// CHECK-H-NEXT: USR:             'B6AC4C5C9F2EA3F2B3ECE1A33D349F4EE502B24E'
// CHECK-H-NEXT: Name:            'H'
// CHECK-H-NEXT: DefLocation:     
// CHECK-H-NEXT:   LineNumber:      223
// CHECK-H-NEXT:   Filename:        'test'
// CHECK-H-NEXT: ReturnType:      
// CHECK-H-NEXT:   Type:            
// CHECK-H-NEXT:     Name:            'void'
// CHECK-H-NEXT: ...
