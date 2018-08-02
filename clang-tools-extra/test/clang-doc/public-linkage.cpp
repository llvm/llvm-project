// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

void function(int x);

inline int inlinedFunction(int x);

int functionWithInnerClass(int x) {
  class InnerClass { //NoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

inline int inlinedFunctionWithInnerClass(int x) {
  class InnerClass { //VisibleNoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

class Class {
public:
  void publicMethod();
  int publicField;

protected:
  void protectedMethod();
  int protectedField;

private:
  void privateMethod();
  int privateField;
};

namespace named {
class NamedClass {
public:
  void namedPublicMethod();
  int namedPublicField;

protected:
  void namedProtectedMethod();
  int namedProtectedField;

private:
  void namedPrivateMethod();
  int namedPrivateField;
};

void namedFunction();
static void namedStaticFunction();
inline void namedInlineFunction();
} // namespace named

static void staticFunction(int x); //Internal Linkage

static int staticFunctionWithInnerClass(int x) {
  class InnerClass { //NoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

namespace {
class AnonClass {
public:
  void anonPublicMethod();
  int anonPublicField;

protected:
  void anonProtectedMethod();
  int anonProtectedField;

private:
  void anonPrivateMethod();
  int anonPrivateField;
};

void anonFunction();
static void anonStaticFunction();
inline void anonInlineFunction();
} // namespace

// RUN: clang-doc --format=yaml --doxygen --public --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./Class.yaml | FileCheck %s --check-prefix CHECK-0
// CHECK-0: ---
// CHECK-0-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT: Name:            'Class'
// CHECK-0-NEXT: DefLocation:     
// CHECK-0-NEXT:   LineNumber:      32
// CHECK-0-NEXT:   Filename:        'test'
// CHECK-0-NEXT: TagType:         Class
// CHECK-0-NEXT: Members:         
// CHECK-0-NEXT:   - Type:            
// CHECK-0-NEXT:       Name:            'int'
// CHECK-0-NEXT:     Name:            'publicField'
// CHECK-0-NEXT:   - Type:            
// CHECK-0-NEXT:       Name:            'int'
// CHECK-0-NEXT:     Name:            'protectedField'
// CHECK-0-NEXT:     Access:          Protected
// CHECK-0-NEXT: ChildFunctions:  
// CHECK-0-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Name:            'publicMethod'
// CHECK-0-NEXT:     Namespace:       
// CHECK-0-NEXT:       - Type:            Record
// CHECK-0-NEXT:         Name:            'Class'
// CHECK-0-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Location:        
// CHECK-0-NEXT:       - LineNumber:      34
// CHECK-0-NEXT:         Filename:        'test'
// CHECK-0-NEXT:     IsMethod:        true
// CHECK-0-NEXT:     Parent:          
// CHECK-0-NEXT:       Type:            Record
// CHECK-0-NEXT:       Name:            'Class'
// CHECK-0-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     ReturnType:      
// CHECK-0-NEXT:       Type:            
// CHECK-0-NEXT:         Name:            'void'
// CHECK-0-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Name:            'protectedMethod'
// CHECK-0-NEXT:     Namespace:       
// CHECK-0-NEXT:       - Type:            Record
// CHECK-0-NEXT:         Name:            'Class'
// CHECK-0-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     Location:        
// CHECK-0-NEXT:       - LineNumber:      38
// CHECK-0-NEXT:         Filename:        'test'
// CHECK-0-NEXT:     IsMethod:        true
// CHECK-0-NEXT:     Parent:          
// CHECK-0-NEXT:       Type:            Record
// CHECK-0-NEXT:       Name:            'Class'
// CHECK-0-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-0-NEXT:     ReturnType:      
// CHECK-0-NEXT:       Type:            
// CHECK-0-NEXT:         Name:            'void'
// CHECK-0-NEXT: ...

// RUN: cat %t/docs/./named.yaml | FileCheck %s --check-prefix CHECK-1
// CHECK-1: ---
// CHECK-1-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-1-NEXT: Name:            'named'
// CHECK-1-NEXT: ChildFunctions:  
// CHECK-1-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-1-NEXT:     Name:            'namedFunction'
// CHECK-1-NEXT:     Namespace:       
// CHECK-1-NEXT:       - Type:            Namespace
// CHECK-1-NEXT:         Name:            'named'
// CHECK-1-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-1-NEXT:     Location:        
// CHECK-1-NEXT:       - LineNumber:      61
// CHECK-1-NEXT:         Filename:        'test'
// CHECK-1-NEXT:     ReturnType:      
// CHECK-1-NEXT:       Type:            
// CHECK-1-NEXT:         Name:            'void'
// CHECK-1-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-1-NEXT:     Name:            'namedInlineFunction'
// CHECK-1-NEXT:     Namespace:       
// CHECK-1-NEXT:       - Type:            Namespace
// CHECK-1-NEXT:         Name:            'named'
// CHECK-1-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-1-NEXT:     Location:        
// CHECK-1-NEXT:       - LineNumber:      63
// CHECK-1-NEXT:         Filename:        'test'
// CHECK-1-NEXT:     ReturnType:      
// CHECK-1-NEXT:       Type:            
// CHECK-1-NEXT:         Name:            'void'
// CHECK-1-NEXT: ...

// RUN: cat %t/docs/./GlobalNamespace.yaml | FileCheck %s --check-prefix CHECK-2
// CHECK-2: ---
// CHECK-2-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT: ChildFunctions:  
// CHECK-2-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT:     Name:            'function'
// CHECK-2-NEXT:     Location:        
// CHECK-2-NEXT:       - LineNumber:      10
// CHECK-2-NEXT:         Filename:        'test'
// CHECK-2-NEXT:     Params:          
// CHECK-2-NEXT:       - Type:            
// CHECK-2-NEXT:           Name:            'int'
// CHECK-2-NEXT:         Name:            'x'
// CHECK-2-NEXT:     ReturnType:      
// CHECK-2-NEXT:       Type:            
// CHECK-2-NEXT:         Name:            'void'
// CHECK-2-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT:     Name:            'inlinedFunction'
// CHECK-2-NEXT:     Location:        
// CHECK-2-NEXT:       - LineNumber:      12
// CHECK-2-NEXT:         Filename:        'test'
// CHECK-2-NEXT:     Params:          
// CHECK-2-NEXT:       - Type:            
// CHECK-2-NEXT:           Name:            'int'
// CHECK-2-NEXT:         Name:            'x'
// CHECK-2-NEXT:     ReturnType:      
// CHECK-2-NEXT:       Type:            
// CHECK-2-NEXT:         Name:            'int'
// CHECK-2-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT:     Name:            'functionWithInnerClass'
// CHECK-2-NEXT:     DefLocation:     
// CHECK-2-NEXT:       LineNumber:      14
// CHECK-2-NEXT:       Filename:        'test'
// CHECK-2-NEXT:     Params:          
// CHECK-2-NEXT:       - Type:            
// CHECK-2-NEXT:           Name:            'int'
// CHECK-2-NEXT:         Name:            'x'
// CHECK-2-NEXT:     ReturnType:      
// CHECK-2-NEXT:       Type:            
// CHECK-2-NEXT:         Name:            'int'
// CHECK-2-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-2-NEXT:     Name:            'inlinedFunctionWithInnerClass'
// CHECK-2-NEXT:     DefLocation:     
// CHECK-2-NEXT:       LineNumber:      23
// CHECK-2-NEXT:       Filename:        'test'
// CHECK-2-NEXT:     Params:          
// CHECK-2-NEXT:       - Type:            
// CHECK-2-NEXT:           Name:            'int'
// CHECK-2-NEXT:         Name:            'x'
// CHECK-2-NEXT:     ReturnType:      
// CHECK-2-NEXT:       Type:            
// CHECK-2-NEXT:         Name:            'int'
// CHECK-2-NEXT: ...

// RUN: cat %t/docs/named/NamedClass.yaml | FileCheck %s --check-prefix CHECK-3
// CHECK-3: ---
// CHECK-3-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT: Name:            'NamedClass'
// CHECK-3-NEXT: Namespace:       
// CHECK-3-NEXT:   - Type:            Namespace
// CHECK-3-NEXT:     Name:            'named'
// CHECK-3-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT: DefLocation:     
// CHECK-3-NEXT:   LineNumber:      47
// CHECK-3-NEXT:   Filename:        'test'
// CHECK-3-NEXT: TagType:         Class
// CHECK-3-NEXT: Members:         
// CHECK-3-NEXT:   - Type:            
// CHECK-3-NEXT:       Name:            'int'
// CHECK-3-NEXT:     Name:            'namedPublicField'
// CHECK-3-NEXT:   - Type:            
// CHECK-3-NEXT:       Name:            'int'
// CHECK-3-NEXT:     Name:            'namedProtectedField'
// CHECK-3-NEXT:     Access:          Protected
// CHECK-3-NEXT: ChildFunctions:  
// CHECK-3-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:     Name:            'namedPublicMethod'
// CHECK-3-NEXT:     Namespace:       
// CHECK-3-NEXT:       - Type:            Record
// CHECK-3-NEXT:         Name:            'NamedClass'
// CHECK-3-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:       - Type:            Namespace
// CHECK-3-NEXT:         Name:            'named'
// CHECK-3-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:     Location:        
// CHECK-3-NEXT:       - LineNumber:      49
// CHECK-3-NEXT:         Filename:        'test'
// CHECK-3-NEXT:     IsMethod:        true
// CHECK-3-NEXT:     Parent:          
// CHECK-3-NEXT:       Type:            Record
// CHECK-3-NEXT:       Name:            'NamedClass'
// CHECK-3-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:     ReturnType:      
// CHECK-3-NEXT:       Type:            
// CHECK-3-NEXT:         Name:            'void'
// CHECK-3-NEXT:   - USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:     Name:            'namedProtectedMethod'
// CHECK-3-NEXT:     Namespace:       
// CHECK-3-NEXT:       - Type:            Record
// CHECK-3-NEXT:         Name:            'NamedClass'
// CHECK-3-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:       - Type:            Namespace
// CHECK-3-NEXT:         Name:            'named'
// CHECK-3-NEXT:         USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:     Location:        
// CHECK-3-NEXT:       - LineNumber:      53
// CHECK-3-NEXT:         Filename:        'test'
// CHECK-3-NEXT:     IsMethod:        true
// CHECK-3-NEXT:     Parent:          
// CHECK-3-NEXT:       Type:            Record
// CHECK-3-NEXT:       Name:            'NamedClass'
// CHECK-3-NEXT:       USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-3-NEXT:     ReturnType:      
// CHECK-3-NEXT:       Type:            
// CHECK-3-NEXT:         Name:            'void'
// CHECK-3-NEXT: ...
