// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

export module M;

int moduleFunction(int x); // ModuleLinkage

static int staticModuleFunction(int x); // ModuleInternalLinkage

export double exportedModuleFunction(double y, int z); // ExternalLinkage

// RUN: clang-doc --format=yaml --doxygen --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./GlobalNamespace.yaml | FileCheck %s --check-prefix CHECK-0
// CHECK-0: ---
// CHECK-0-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT: ChildFunctions:  
// CHECK-0-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Name:            'moduleFunction'
// CHECK-0-NEXT:     Location:        
// CHECK-0-NEXT:       - LineNumber:      11
// CHECK-0-NEXT:         Filename:        'test'
// CHECK-0-NEXT:     Params:          
// CHECK-0-NEXT:       - Type:            
// CHECK-0-NEXT:           Name:            'int'
// CHECK-0-NEXT:         Name:            'x'
// CHECK-0-NEXT:     ReturnType:      
// CHECK-0-NEXT:       Type:            
// CHECK-0-NEXT:         Name:            'int'
// CHECK-0-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Name:            'staticModuleFunction'
// CHECK-0-NEXT:     Location:        
// CHECK-0-NEXT:       - LineNumber:      13
// CHECK-0-NEXT:         Filename:        'test'
// CHECK-0-NEXT:     Params:          
// CHECK-0-NEXT:       - Type:            
// CHECK-0-NEXT:           Name:            'int'
// CHECK-0-NEXT:         Name:            'x'
// CHECK-0-NEXT:     ReturnType:      
// CHECK-0-NEXT:       Type:            
// CHECK-0-NEXT:         Name:            'int'
// CHECK-0-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Name:            'exportedModuleFunction'
// CHECK-0-NEXT:     Location:        
// CHECK-0-NEXT:       - LineNumber:      15
// CHECK-0-NEXT:         Filename:        'test'
// CHECK-0-NEXT:     Params:          
// CHECK-0-NEXT:       - Type:            
// CHECK-0-NEXT:           Name:            'double'
// CHECK-0-NEXT:         Name:            'y'
// CHECK-0-NEXT:       - Type:            
// CHECK-0-NEXT:           Name:            'int'
// CHECK-0-NEXT:         Name:            'z'
// CHECK-0-NEXT:     ReturnType:      
// CHECK-0-NEXT:       Type:            
// CHECK-0-NEXT:         Name:            'double'
// CHECK-0-NEXT: ...
