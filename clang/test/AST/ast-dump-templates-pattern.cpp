// RUN: %clang_cc1 -triple x86_64 -std=c++26 -ast-dump -ast-dump-filter=Test %s \
// RUN: | FileCheck --match-full-lines --check-prefixes CHECK,CHECK-SRC %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64 -std=c++26 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64 -x c++ -std=c++26 -include-pch %t -ast-dump-all -ast-dump-filter=Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --match-full-lines --check-prefixes CHECK,CHECK-PCH %s

namespace TestClassRedecl {
  template <class T> struct A {};
  template <class T> struct A;
  template struct A<int>;
// CHECK-LABEL: Dumping TestClassRedecl:
// CHECK: |-ClassTemplateDecl {{.+}} <line:[[@LINE-4]]:{{.+}} A external-linkage
// CHECK: | |-CXXRecordDecl 0x[[TestClassRedecl_D1:[^ ]+]] {{.+}} struct A definition
// CHECK: | `-ClassTemplateSpecialization 0x[[TestClassRedecl_S:[^ ]+]] 'A'
// CHECK: |-ClassTemplateDecl {{.+}} <line:[[@LINE-6]]:{{.+}} A  external-linkage
// CHECK: | |-CXXRecordDecl 0x[[TestClassRedecl_D2:[^ ]+]] prev 0x[[TestClassRedecl_D1]] {{.+}} struct A
// CHECK: | `-ClassTemplateSpecialization 0x[[TestClassRedecl_S]] 'A'
// CHECK: |-ClassTemplateSpecializationDecl 0x[[TestClassRedecl_S]] <line:[[@LINE-8]]:{{.+}} struct A definition external-linkage instantiated_from 0x[[TestClassRedecl_D2]] explicit_instantiation_definition
// CHECK: `-ExplicitInstantiationDecl {{.+}} <line:[[@LINE-9]]:{{.+}} 'A'
// CHECK:   |-ClassTemplateSpecialization 0x[[TestClassRedecl_S]] 'A'
// CHECK:   `-TemplateArgument {{.+}} type 'int'
// CHECK:     `-BuiltinType {{.+}} 'int'
}

namespace TestFunctionRedecl {
  template <class T> void f() {}
  template <class T> void f();
  template void f<int>();
// CHECK-LABEL: Dumping TestFunctionRedecl:
// CHECK: |-FunctionTemplateDecl 0x[[TestFunctionRedecl_T1:[^ ]+]] <line:[[@LINE-4]]:{{.+}} f external-linkage
// CHECK: | |-FunctionDecl 0x[[TestFunctionRedecl_D1:[^ ]+]] {{.+}} f 'void ()'
// CHECK: | |-FunctionDecl {{.+}} prev 0x[[TestFunctionRedecl_S1:[^ ]+]] <col:22, col:32> col:27 f 'void ()' explicit_instantiation_definition instantiated_from 0x[[TestFunctionRedecl_D1]] external-linkage
// CHECK: | `-FunctionDecl 0x[[TestFunctionRedecl_S1]] {{.+}} f 'void ()' explicit_instantiation_definition instantiated_from 0x[[TestFunctionRedecl_D2:[^ ]+]] external-linkage
// CHECK: |-FunctionTemplateDecl 0x[[TestFunctionRedecl_T2:[^ ]+]] prev 0x[[TestFunctionRedecl_T1]] <line:[[@LINE-7]]:{{.+}} f external-linkage
// CHECK: | |-FunctionDecl 0x[[TestFunctionRedecl_D2]] prev 0x[[TestFunctionRedecl_D1]] {{.+}} f 'void ()'
// CHECK: | `-Function 0x[[TestFunctionRedecl_S1]] 'f' 'void ()'
// CHECK: `-ExplicitInstantiationDecl {{.+}} <line:[[@LINE-9]]:{{.+}} 'f'
// CHECK:   |-Function 0x[[TestFunctionRedecl_S1]] 'f' 'void ()'
// CHECK:   |-FunctionProtoTypeLoc {{.+}} 'void ()' cdecl
// CHECK:   | `-BuiltinTypeLoc {{.+}} 'void'
// CHECK:   `-TemplateArgument {{.+}} type 'int'
// CHECK:     `-BuiltinType {{.+}} 'int'
}

namespace TestVariableRedecl {
  template <class T> T a = 0;
  template <class T> extern T a;
  template int a<int>;
// CHECK-LABEL: Dumping TestVariableRedecl:
// CHECK: |-VarTemplateDecl 0x[[TestVariableRedecl_T1:[^ ]+]] <line:[[@LINE-4]]:{{.+}} a external-linkage
// CHECK: | |-VarDecl 0x[[TestVariableRedecl_D1:[^ ]+]] {{.+}} a 'T' cinit{{$}}
// CHECK: | `-VarTemplateSpecialization 0x[[TestVariableRedecl_S1:[^ ]+]] 'a' 'int'
// CHECK: |-VarTemplateDecl 0x[[TestVariableRedecl_T2:[^ ]+]] prev 0x[[TestVariableRedecl_T1]] <line:[[@LINE-6]]:{{.+}} a external-linkage
// CHECK: | |-VarDecl 0x[[TestVariableRedecl_D2:[^ ]+]] prev 0x[[TestVariableRedecl_D1]] {{.+}} a 'T' extern{{$}}
// CHECK: | `-VarTemplateSpecialization 0x[[TestVariableRedecl_S1]] 'a' 'int'
// CHECK: |-VarTemplateSpecializationDecl 0x[[TestVariableRedecl_S1]] {{.+}} a 'int' explicit_instantiation_definition cinit instantiated_from 0x[[TestVariableRedecl_D2]] external-linkage
// CHECK: `-ExplicitInstantiationDecl {{.+}} <line:[[@LINE-9]]:{{.+}} 'a'
// CHECK:   |-VarTemplateSpecialization 0x[[TestVariableRedecl_S1]] 'a' 'int'
// CHECK:   |-BuiltinTypeLoc {{.+}} 'int'
// CHECK:   `-TemplateArgument {{.+}} type 'int'
// CHECK:     `-BuiltinType {{.+}} 'int'
}

namespace TestNestedClassRedecl {
  template <class T> struct A {
    template <class U> struct B;
  };
  template <class T> struct A;
  template <class T> template <class U> struct A<T>::B {};
  template struct A<int>::B<char>;
// FIXME: Serialization doesn't preserve exact parent because DeclContext merging doesn't discriminate redeclarations properly.
// CHECK-LABEL: Dumping TestNestedClassRedecl:
// CHECK: |-ClassTemplateDecl 0x[[TestNestedClassRedecl_A_T1:[^ ]+]] <line:[[@LINE-8]]:{{.+}} A external-linkage
// CHECK: | |-CXXRecordDecl 0x[[TestNestedClassRedecl_A_D1:[^ ]+]] <{{.+}}line:[[@LINE-7]]:{{.+}}> line:[[@LINE-9]]:{{.+}} struct A definition
// CHECK: | | `-ClassTemplateDecl 0x[[TestNestedClassRedecl_B_T1:[^ ]+]] <line:[[@LINE-9]]:{{.+}} B external-linkage
// CHECK: | |   `-CXXRecordDecl 0x[[TestNestedClassRedecl_B_D1:[^ ]+]] {{.+}} struct B
// CHECK: | `-ClassTemplateSpecializationDecl 0x[[TestNestedClassRedecl_A_S1:[^ ]+]] <line:[[@LINE-9]]:{{.+}} line:[[@LINE-12]]:{{.+}} struct A definition external-linkage instantiated_from 0x[[TestNestedClassRedecl_A_D2:[^ ]+]] implicit_instantiation
// CHECK: |   |-ClassTemplateDecl 0x{{.+}} <line:[[@LINE-12]]:{{.+}} B external-linkage
// CHECK: |   | |-CXXRecordDecl 0x{{.+}} struct B
// CHECK: |   | `-ClassTemplateSpecialization  0x[[TestNestedClassRedecl_B_S1:[^ ]+]] 'B'
// CHECK: |   `-ClassTemplateSpecializationDecl 0x[[TestNestedClassRedecl_B_S1]] <line:[[@LINE-11]]:{{.+}} struct B definition external-linkage instantiated_from 0x[[TestNestedClassRedecl_B_D1]] explicit_instantiation_definition
// CHECK: |-ClassTemplateDecl 0x{{.+}} prev 0x[[TestNestedClassRedecl_A_T1]] <line:[[@LINE-14]]:{{.+}} A external-linkage
// CHECK: | |-CXXRecordDecl 0x[[TestNestedClassRedecl_A_D2]] prev 0x[[TestNestedClassRedecl_A_D1]] {{.+}} struct A
// CHECK: | `-ClassTemplateSpecialization 0x[[TestNestedClassRedecl_A_S1]] 'A'
// CHECK-SRC: |-ClassTemplateDecl 0x{{.+}} parent 0x[[TestNestedClassRedecl_A_D2]] prev 0x[[TestNestedClassRedecl_B_T1]] <line:[[@LINE-16]]:{{.+}} B external-linkage
// CHECK-SRC: | `-CXXRecordDecl 0x[[TestNestedClassRedecl_B_D2:[^ ]+]] parent 0x[[TestNestedClassRedecl_A_D2]] prev 0x[[TestNestedClassRedecl_B_D1]] {{.+}} struct B definition
// CHECK-PCH: |-ClassTemplateDecl 0x{{.+}} parent 0x[[TestNestedClassRedecl_A_D1]] prev 0x[[TestNestedClassRedecl_B_T1]] <line:[[@LINE-18]]:{{.+}} B external-linkage
// CHECK-PCH: | `-CXXRecordDecl 0x[[TestNestedClassRedecl_B_D2:[^ ]+]] parent 0x[[TestNestedClassRedecl_A_D1]] prev 0x[[TestNestedClassRedecl_B_D1]] {{.+}} struct B definition
// CHECK: `-ExplicitInstantiationDecl {{.+}} <line:[[@LINE-19]]:{{.+}} 'B'
// CHECK:   |-NestedNameSpecifier TypeSpec {{.+}}
// CHECK:   |-ClassTemplateSpecialization 0x[[TestNestedClassRedecl_B_S1]] 'B'
// CHECK:   `-TemplateArgument {{.+}} type 'char'
// CHECK:     `-BuiltinType {{.+}} 'char'
}
