// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

namespace ns {
  template <typename T> void foo(T x) {}
  template <typename T> T bar = T{};
  template <typename T> struct S {
    void method(T x) {}
    static T sval;
    static T arr[1];
    template <typename U> void tmpl(U u) {}
    template <typename U> static U mvar;
    template <typename U> struct Nested {};
    struct Inner { T val; };
  };
  template <typename T> T S<T>::sval = T{};
  template <typename T> T S<T>::arr[1] = {};
  template <typename T> template <typename U> U S<T>::mvar = U{};

  template <typename T> struct A {
    template <typename U> struct B {
      template <typename V> void deep(V v) {}
    };
  };
}

// (a) function template
template void ns::foo<int>(int);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:31> col:1 explicit_instantiation_definition 'foo'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: Function {{.*}} 'foo' 'void (int)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:10, col:31> 'void (int)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:28> col:31 'int'
// CHECK-NEXT:     BuiltinTypeLoc <col:28> 'int'
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'void'
// CHECK-NEXT: TemplateArgument <col:23> type 'int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

// (b) variable template
template int ns::bar<int>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:25> col:1 explicit_instantiation_definition 'bar'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'int'
// CHECK-NEXT: BuiltinTypeLoc <col:10> 'int'
// CHECK-NEXT: TemplateArgument <col:22> type 'int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

// (c) class template
template struct ns::S<int>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:26> col:1 explicit_instantiation_definition 'S'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: ClassTemplateSpecialization {{.*}} 'S'
// CHECK-NEXT: TemplateArgument <col:23> type 'int'
// CHECK-NEXT:   BuiltinType {{.*}} 'int'

// (d) member function
template void ns::S<long>::method(long);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:39> col:1 explicit_instantiation_definition 'method'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: CXXMethod {{.*}} 'method' 'void (long)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:10, col:39> 'void (long)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:35> col:39 'long'
// CHECK-NEXT:     BuiltinTypeLoc <col:35> 'long'
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'void'

// (e) static data member
template long ns::S<long>::sval;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:28> col:1 explicit_instantiation_definition 'sval'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: Var {{.*}} 'sval' 'long'
// CHECK-NEXT: BuiltinTypeLoc <col:10> 'long'

// (e2) static data member with postfix type (array)
template long ns::S<long>::arr[1];
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:33> col:1 explicit_instantiation_definition 'arr'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: Var {{.*}} 'arr' 'long[1]'
// CHECK-NEXT: ConstantArrayTypeLoc <col:10, col:33> 'long[1]' 1
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'long'

// (f) member function template
template void ns::S<long>::tmpl<double>(double);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:47> col:1 explicit_instantiation_definition 'tmpl'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: CXXMethod {{.*}} 'tmpl' 'void (double)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:10, col:47> 'void (double)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:41> col:47 'double'
// CHECK-NEXT:     BuiltinTypeLoc <col:41> 'double'
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'void'
// CHECK-NEXT: TemplateArgument <col:33> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

// (g) nested class
template struct ns::S<long>::Inner;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:30> col:1 explicit_instantiation_definition 'Inner'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: CXXRecord {{.*}} 'Inner'

// extern template variants
extern template void ns::foo<float>(float);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:42> col:8 explicit_instantiation_declaration extern 'foo'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: Function {{.*}} 'foo' 'void (float)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:17, col:42> 'void (float)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:37> col:42 'float'
// CHECK-NEXT:     BuiltinTypeLoc <col:37> 'float'
// CHECK-NEXT:   BuiltinTypeLoc <col:17> 'void'
// CHECK-NEXT: TemplateArgument <col:30> type 'float'
// CHECK-NEXT:   BuiltinType {{.*}} 'float'

extern template struct ns::S<float>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:35> col:8 explicit_instantiation_declaration extern 'S'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: ClassTemplateSpecialization {{.*}} 'S'
// CHECK-NEXT: TemplateArgument <col:30> type 'float'
// CHECK-NEXT:   BuiltinType {{.*}} 'float'

// extern template: variable template
extern template double ns::bar<double>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:38> col:8 explicit_instantiation_declaration extern 'bar'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'double'
// CHECK-NEXT: BuiltinTypeLoc <col:17> 'double'
// CHECK-NEXT: TemplateArgument <col:32> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

// extern template: member function
extern template void ns::S<double>::method(double);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:50> col:8 explicit_instantiation_declaration extern 'method'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: CXXMethod {{.*}} 'method' 'void (double)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:17, col:50> 'void (double)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:44> col:50 'double'
// CHECK-NEXT:     BuiltinTypeLoc <col:44> 'double'
// CHECK-NEXT:   BuiltinTypeLoc <col:17> 'void'

// extern template: static data member
extern template double ns::S<double>::sval;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:39> col:8 explicit_instantiation_declaration extern 'sval'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: Var {{.*}} 'sval' 'double'
// CHECK-NEXT: BuiltinTypeLoc <col:17> 'double'

// extern template: member function template
extern template void ns::S<double>::tmpl<float>(float);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:54> col:8 explicit_instantiation_declaration extern 'tmpl'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: CXXMethod {{.*}} 'tmpl' 'void (float)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:17, col:54> 'void (float)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:49> col:54 'float'
// CHECK-NEXT:     BuiltinTypeLoc <col:49> 'float'
// CHECK-NEXT:   BuiltinTypeLoc <col:17> 'void'
// CHECK-NEXT: TemplateArgument <col:42> type 'float'
// CHECK-NEXT:   BuiltinType {{.*}} 'float'

// extern template: nested class
extern template struct ns::S<double>::Inner;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:39> col:8 explicit_instantiation_declaration extern 'Inner'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: CXXRecord {{.*}} 'Inner'

// member variable template
template double ns::S<long>::mvar<double>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:41> col:1 explicit_instantiation_definition 'mvar'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'mvar' 'double'
// CHECK-NEXT: BuiltinTypeLoc <col:10> 'double'
// CHECK-NEXT: TemplateArgument <col:35> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

// member class template
template struct ns::S<long>::Nested<double>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:43> col:1 explicit_instantiation_definition 'Nested'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: ClassTemplateSpecialization {{.*}} 'Nested'
// CHECK-NEXT: TemplateArgument <col:37> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

// deeply nested: A<int>::B<double>::deep<float>
template void ns::A<int>::B<double>::deep<float>(float);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:55> col:1 explicit_instantiation_definition 'deep'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::A<int>::B<double>'
// CHECK-NEXT: CXXMethod {{.*}} 'deep' 'void (float)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:10, col:55> 'void (float)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:50> col:55 'float'
// CHECK-NEXT:     BuiltinTypeLoc <col:50> 'float'
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'void'
// CHECK-NEXT: TemplateArgument <col:43> type 'float'
// CHECK-NEXT:   BuiltinType {{.*}} 'float'

// Same-namespace explicit instantiation (no cross-namespace qualifier)
namespace ns {
  template void foo<short>(short);
  // CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:3, col:33> col:3 explicit_instantiation_definition 'foo'
  // CHECK-NEXT: Function {{.*}} 'foo' 'void (short)'
  // CHECK-NEXT: FunctionProtoTypeLoc <col:12, col:33> 'void (short)' cdecl
  // CHECK-NEXT:   ParmVarDecl {{.*}} <col:28> col:33 'short'
  // CHECK-NEXT:     BuiltinTypeLoc <col:28> 'short'
  // CHECK-NEXT:   BuiltinTypeLoc <col:12> 'void'
  // CHECK-NEXT: TemplateArgument <col:21> type 'short'
  // CHECK-NEXT:   BuiltinType {{.*}} 'short'

  template short bar<short>;
  // CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:3, col:27> col:3 explicit_instantiation_definition 'bar'
  // CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'short'
  // CHECK-NEXT: BuiltinTypeLoc <col:12> 'short'
  // CHECK-NEXT: TemplateArgument <col:22> type 'short'
  // CHECK-NEXT:   BuiltinType {{.*}} 'short'

  template struct S<short>;
  // CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:3, col:26> col:3 explicit_instantiation_definition 'S'
  // CHECK-NEXT: ClassTemplateSpecialization {{.*}} 'S'
  // CHECK-NEXT: TemplateArgument <col:21> type 'short'
  // CHECK-NEXT:   BuiltinType {{.*}} 'short'
}
