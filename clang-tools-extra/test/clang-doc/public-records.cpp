// This test requires linux because it uses `diff` and compares filepaths
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --public --doxygen -p %t %t/test.cpp -output=%t/docs
// RUN: clang-doc --doxygen -p %t %t/test.cpp -output=%t/docs-without-flag
// RUN: cat %t/docs/function.yaml | FileCheck %s --check-prefix=CHECK-A
// RUN: cat %t/docs/inlinedFunction.yaml | FileCheck %s --check-prefix=CHECK-B
// RUN: cat %t/docs/functionWithInnerClass.yaml | FileCheck %s --check-prefix=CHECK-C
// RUN: cat %t/docs/inlinedFunctionWithInnerClass.yaml | FileCheck %s --check-prefix=CHECK-D
// RUN: cat %t/docs/Class/publicMethod.yaml| FileCheck %s --check-prefix=CHECK-E
// RUN: cat %t/docs/Class.yaml| FileCheck %s --check-prefix=CHECK-F
// RUN: cat %t/docs/Class/protectedMethod.yaml| FileCheck %s --check-prefix=CHECK-G
// RUN: cat %t/docs/named.yaml| FileCheck %s --check-prefix=CHECK-H
// RUN: cat %t/docs/named/NamedClass.yaml| FileCheck %s --check-prefix=CHECK-I
// RUN: cat %t/docs/named/namedFunction.yaml| FileCheck %s --check-prefix=CHECK-J
// RUN: cat %t/docs/named/namedInlineFunction.yaml| FileCheck %s --check-prefix=CHECK-K
// RUN: cat %t/docs/named/NamedClass/namedPublicMethod.yaml| FileCheck %s --check-prefix=CHECK-L
// RUN: cat %t/docs/named/NamedClass/namedProtectedMethod.yaml| FileCheck %s --check-prefix=CHECK-M
// RUN: (diff -qry %t/docs-without-flag %t/docs | sed 's:.*/::' > %t/public.diff) || true
// RUN: cat %t/public.diff | FileCheck %s --check-prefix=CHECK-N

void function(int x);

// CHECK-A: ---
// CHECK-A-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-A-NEXT: Name:            'function'
// CHECK-A-NEXT: Location:
// CHECK-A-NEXT:   - LineNumber:      25
// CHECK-A-NEXT:     Filename:        {{.*}}
// CHECK-A-NEXT: Params:
// CHECK-A-NEXT:   - Type:
// CHECK-A-NEXT:       Name:            'int'
// CHECK-A-NEXT:     Name:            'x'
// CHECK-A-NEXT: ReturnType:
// CHECK-A-NEXT:   Type:
// CHECK-A-NEXT:     Name:            'void'
// CHECK-A-NEXT: ...

inline int inlinedFunction(int x);

// CHECK-B: ---
// CHECK-B-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-B-NEXT: Name:            'inlinedFunction'
// CHECK-B-NEXT: Location:
// CHECK-B-NEXT:   - LineNumber:      42
// CHECK-B-NEXT:     Filename:        {{.*}}
// CHECK-B-NEXT: Params:
// CHECK-B-NEXT:   - Type:
// CHECK-B-NEXT:       Name:            'int'
// CHECK-B-NEXT:     Name:            'x'
// CHECK-B-NEXT: ReturnType:
// CHECK-B-NEXT:   Type:
// CHECK-B-NEXT:     Name:            'int'
// CHECK-B-NEXT: ...

int functionWithInnerClass(int x){
    class InnerClass { //NoLinkage
      public:
        int innerPublicMethod() { return 2; };
    }; //end class
    InnerClass temp;
    return temp.innerPublicMethod();
};

// CHECK-C: ---
// CHECK-C-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-C-NEXT: Name:            'functionWithInnerClass'
// CHECK-C-NEXT: DefLocation:
// CHECK-C-NEXT:   LineNumber:      59
// CHECK-C-NEXT:   Filename:        {{.*}}
// CHECK-C-NEXT: Params:
// CHECK-C-NEXT:   - Type:
// CHECK-C-NEXT:       Name:            'int'
// CHECK-C-NEXT:     Name:            'x'
// CHECK-C-NEXT: ReturnType:
// CHECK-C-NEXT:   Type:
// CHECK-C-NEXT:     Name:            'int'
// CHECK-C-NEXT: ...

inline int inlinedFunctionWithInnerClass(int x){
    class InnerClass { //VisibleNoLinkage
      public:
        int innerPublicMethod() { return 2; };
    }; //end class
    InnerClass temp;
    return temp.innerPublicMethod();
};

// CHECK-D: ---
// CHECK-D-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-D-NEXT: Name:            'inlinedFunctionWithInnerClass'
// CHECK-D-NEXT: DefLocation:
// CHECK-D-NEXT:   LineNumber:      83
// CHECK-D-NEXT:   Filename:        {{.*}}
// CHECK-D-NEXT: Params:
// CHECK-D-NEXT:   - Type:
// CHECK-D-NEXT:       Name:            'int'
// CHECK-D-NEXT:     Name:            'x'
// CHECK-D-NEXT: ReturnType:
// CHECK-D-NEXT:   Type:
// CHECK-D-NEXT:     Name:            'int'
// CHECK-D-NEXT: ...

class Class {
 public:
  void publicMethod();
  int  publicField;
 protected:
  void protectedMethod();
  int  protectedField;
 private:
  void privateMethod();
  int  privateField;
};

// CHECK-E: ---
// CHECK-E-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-E-NEXT: Name:            'publicMethod'
// CHECK-E-NEXT: Namespace:
// CHECK-E-NEXT:   - Type:            Record
// CHECK-E-NEXT:     Name:            'Class'
// CHECK-E-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-E-NEXT: Location:
// CHECK-E-NEXT:   - LineNumber:      109
// CHECK-E-NEXT:     Filename:        {{.*}}
// CHECK-E-NEXT: IsMethod:        true
// CHECK-E-NEXT: Parent:
// CHECK-E-NEXT:   Type:            Record
// CHECK-E-NEXT:   Name:            'Class'
// CHECK-E-NEXT:   USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-E-NEXT: ReturnType:
// CHECK-E-NEXT:   Type:
// CHECK-E-NEXT:     Name:            'void'
// CHECK-E-NEXT: ...

// CHECK-F: ---
// CHECK-F-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-F-NEXT: Name:            'Class'
// CHECK-F-NEXT: DefLocation:
// CHECK-F-NEXT:   LineNumber:      107
// CHECK-F-NEXT:   Filename:        {{.*}}
// CHECK-F-NEXT: TagType:         Class
// CHECK-F-NEXT: Members:
// CHECK-F-NEXT:   - Type:
// CHECK-F-NEXT:       Name:            'int'
// CHECK-F-NEXT:     Name:            'publicField'
// CHECK-F-NEXT:   - Type:
// CHECK-F-NEXT:       Name:            'int'
// CHECK-F-NEXT:     Name:            'protectedField'
// CHECK-F-NEXT:     Access:          Protected
// CHECK-F-NEXT: ...

// CHECK-G: ---
// CHECK-G-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-G-NEXT: Name:            'protectedMethod'
// CHECK-G-NEXT: Namespace:
// CHECK-G-NEXT:   - Type:            Record
// CHECK-G-NEXT:     Name:            'Class'
// CHECK-G-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-G-NEXT: Location:
// CHECK-G-NEXT:   - LineNumber:      112
// CHECK-G-NEXT:     Filename:        {{.*}}
// CHECK-G-NEXT: IsMethod:        true
// CHECK-G-NEXT: Parent:
// CHECK-G-NEXT:   Type:            Record
// CHECK-G-NEXT:   Name:            'Class'
// CHECK-G-NEXT:   USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-G-NEXT: ReturnType:
// CHECK-G-NEXT:   Type:
// CHECK-G-NEXT:     Name:            'void'
// CHECK-G-NEXT: ...

namespace named{
    class NamedClass {
     public:
      void namedPublicMethod();
      int  namedPublicField;
     protected:
      void namedProtectedMethod();
      int  namedProtectedField;
     private:
      void namedPrivateMethod();
      int  namedPrivateField;
    };

    void namedFunction();
    static void namedStaticFunction();
    inline void namedInlineFunction();
}

// CHECK-H: ---
// CHECK-H-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-H-NEXT: Name:            'named'
// CHECK-H-NEXT: ...

// CHECK-I: ---
// CHECK-I-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-I-NEXT: Name:            'NamedClass'
// CHECK-I-NEXT: Namespace:
// CHECK-I-NEXT:   - Type:            Namespace
// CHECK-I-NEXT:     Name:            'named'
// CHECK-I-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-I-NEXT: DefLocation:
// CHECK-I-NEXT:   LineNumber:      177
// CHECK-I-NEXT:   Filename:        {{.*}}
// CHECK-I-NEXT: TagType:         Class
// CHECK-I-NEXT: Members:
// CHECK-I-NEXT:   - Type:
// CHECK-I-NEXT:       Name:            'int'
// CHECK-I-NEXT:     Name:            'namedPublicField'
// CHECK-I-NEXT:   - Type:
// CHECK-I-NEXT:       Name:            'int'
// CHECK-I-NEXT:     Name:            'namedProtectedField'
// CHECK-I-NEXT:     Access:          Protected
// CHECK-I-NEXT: ...

// CHECK-J: ---
// CHECK-J-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-J-NEXT: Name:            'namedFunction'
// CHECK-J-NEXT: Namespace:
// CHECK-J-NEXT:   - Type:            Namespace
// CHECK-J-NEXT:     Name:            'named'
// CHECK-J-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-J-NEXT: Location:
// CHECK-J-NEXT:   - LineNumber:      189
// CHECK-J-NEXT:     Filename:        {{.*}}
// CHECK-J-NEXT: ReturnType:
// CHECK-J-NEXT:   Type:
// CHECK-J-NEXT:     Name:            'void'
// CHECK-J-NEXT: ...

// CHECK-K: ---
// CHECK-K-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-K-NEXT: Name:            'namedInlineFunction'
// CHECK-K-NEXT: Namespace:
// CHECK-K-NEXT:   - Type:            Namespace
// CHECK-K-NEXT:     Name:            'named'
// CHECK-K-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-K-NEXT: Location:
// CHECK-K-NEXT:   - LineNumber:      191
// CHECK-K-NEXT:     Filename:        {{.*}}
// CHECK-K-NEXT: ReturnType:
// CHECK-K-NEXT:   Type:
// CHECK-K-NEXT:     Name:            'void'
// CHECK-K-NEXT: ...

// CHECK-L: ---
// CHECK-L-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-L-NEXT: Name:            'namedPublicMethod'
// CHECK-L-NEXT: Namespace:
// CHECK-L-NEXT:   - Type:            Record
// CHECK-L-NEXT:     Name:            'NamedClass'
// CHECK-L-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-L-NEXT:   - Type:            Namespace
// CHECK-L-NEXT:     Name:            'named'
// CHECK-L-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-L-NEXT: Location:
// CHECK-L-NEXT:   - LineNumber:      179
// CHECK-L-NEXT:     Filename:        {{.*}}
// CHECK-L-NEXT: IsMethod:        true
// CHECK-L-NEXT: Parent:
// CHECK-L-NEXT:   Type:            Record
// CHECK-L-NEXT:   Name:            'NamedClass'
// CHECK-L-NEXT:   USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-L-NEXT: ReturnType:
// CHECK-L-NEXT:   Type:
// CHECK-L-NEXT:     Name:            'void'
// CHECK-L-NEXT: ...

// CHECK-M: ---
// CHECK-M-NEXT: USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-M-NEXT: Name:            'namedProtectedMethod'
// CHECK-M-NEXT: Namespace:
// CHECK-M-NEXT:   - Type:            Record
// CHECK-M-NEXT:     Name:            'NamedClass'
// CHECK-M-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-M-NEXT:   - Type:            Namespace
// CHECK-M-NEXT:     Name:            'named'
// CHECK-M-NEXT:     USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-M-NEXT: Location:
// CHECK-M-NEXT:   - LineNumber:      182
// CHECK-M-NEXT:     Filename:        {{.*}}
// CHECK-M-NEXT: IsMethod:        true
// CHECK-M-NEXT: Parent:
// CHECK-M-NEXT:   Type:            Record
// CHECK-M-NEXT:   Name:            'NamedClass'
// CHECK-M-NEXT:   USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'
// CHECK-M-NEXT: ReturnType:
// CHECK-M-NEXT:   Type:
// CHECK-M-NEXT:     Name:            'void'
// CHECK-M-NEXT: ...


static void staticFunction(int x); //Internal Linkage

static int staticFunctionWithInnerClass(int x){
    class InnerClass { //NoLinkage
      public:
        int innerPublicMethod() { return 2; };
    }; //end class
    InnerClass temp;
    return temp.innerPublicMethod();
};

namespace{
    class AnonClass {
     public:
      void anonPublicMethod();
      int  anonPublicField;
     protected:
      void anonProtectedMethod();
      int  anonProtectedField;
     private:
      void anonPrivateMethod();
      int  anonPrivateField;
    };

    void anonFunction();
    static void anonStaticFunction();
    inline void anonInlineFunction();
}

// CHECK-N: docs-without-flag: .yaml
// CHECK-N-NEXT: docs-without-flag: AnonClass
// CHECK-N-NEXT: docs-without-flag: AnonClass.yaml
// CHECK-N-NEXT: Class: privateMethod.yaml
// CHECK-N-NEXT: Class.yaml differ
// CHECK-N-NEXT: docs-without-flag: anonFunction.yaml
// CHECK-N-NEXT: docs-without-flag: anonInlineFunction.yaml
// CHECK-N-NEXT: docs-without-flag: anonStaticFunction.yaml
// CHECK-N-NEXT: docs-without-flag: functionWithInnerClass
// CHECK-N-NEXT: docs-without-flag: inlinedFunctionWithInnerClass
// CHECK-N-NEXT: NamedClass: namedPrivateMethod.yaml
// CHECK-N-NEXT: NamedClass.yaml differ
// CHECK-N-NEXT: named: namedStaticFunction.yaml
// CHECK-N-NEXT: docs-without-flag: staticFunction.yaml
// CHECK-N-NEXT: docs-without-flag: staticFunctionWithInnerClass
// CHECK-N-NEXT: docs-without-flag: staticFunctionWithInnerClass.yaml
