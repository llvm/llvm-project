// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

struct Plain {};
enum Color { Red };
template <typename T> struct Wrap {};
template <typename T = int> struct Defaulted {};
template <typename T = int> T defaulted_var = T{};

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

// extern template: member variable template
extern template double ns::S<double>::mvar<double>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:50> col:8 explicit_instantiation_declaration extern 'mvar'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'mvar' 'double'
// CHECK-NEXT: BuiltinTypeLoc <col:17> 'double'
// CHECK-NEXT: TemplateArgument <col:44> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

// extern template: member class template
extern template struct ns::S<double>::Nested<double>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:52> col:8 explicit_instantiation_declaration extern 'Nested'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: ClassTemplateSpecialization {{.*}} 'Nested'
// CHECK-NEXT: TemplateArgument <col:46> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

// extern template: static data member (array)
extern template double ns::S<double>::arr[1];
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:44> col:8 explicit_instantiation_declaration extern 'arr'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<double>'
// CHECK-NEXT: Var {{.*}} 'arr' 'double[1]'
// CHECK-NEXT: ConstantArrayTypeLoc <col:17, col:44> 'double[1]' 1
// CHECK-NEXT:   BuiltinTypeLoc <col:17> 'double'

// extern template: deeply nested
extern template void ns::A<int>::B<float>::deep<double>(double);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:63> col:8 explicit_instantiation_declaration extern 'deep'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::A<int>::B<float>'
// CHECK-NEXT: CXXMethod {{.*}} 'deep' 'void (double)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:17, col:63> 'void (double)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:57> col:63 'double'
// CHECK-NEXT:     BuiltinTypeLoc <col:57> 'double'
// CHECK-NEXT:   BuiltinTypeLoc <col:17> 'void'
// CHECK-NEXT: TemplateArgument <col:49> type 'double'
// CHECK-NEXT:   BuiltinType {{.*}} 'double'

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

// empty <> (args deduced)
template void ns::foo<>(double);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:31> col:1 explicit_instantiation_definition 'foo'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: Function {{.*}} 'foo' 'void (double)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:10, col:31> 'void (double)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:25> col:31 'double'
// CHECK-NEXT:     BuiltinTypeLoc <col:25> 'double'
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'void'

template void ns::S<long>::tmpl<>(float);
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:40> col:1 explicit_instantiation_definition 'tmpl'
// CHECK-NEXT: NestedNameSpecifier TypeSpec 'ns::S<long>'
// CHECK-NEXT: CXXMethod {{.*}} 'tmpl' 'void (float)'
// CHECK-NEXT: FunctionProtoTypeLoc <col:10, col:40> 'void (float)' cdecl
// CHECK-NEXT:   ParmVarDecl {{.*}} <col:35> col:40 'float'
// CHECK-NEXT:     BuiltinTypeLoc <col:35> 'float'
// CHECK-NEXT:   BuiltinTypeLoc <col:10> 'void'

// empty <> (default template arguments)
template struct Defaulted<>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:27> col:1 explicit_instantiation_definition 'Defaulted'
// CHECK-NEXT: ClassTemplateSpecialization {{.*}} 'Defaulted'

template int defaulted_var<>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:28> col:1 explicit_instantiation_definition 'defaulted_var'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'defaulted_var' 'int'
// CHECK-NEXT: BuiltinTypeLoc <col:10> 'int'

extern template Plain ns::bar<Plain>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:36> col:8 explicit_instantiation_declaration extern 'bar'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'Plain'
// CHECK-NEXT: RecordTypeLoc <col:17> 'Plain'
// CHECK:      TemplateArgument <col:31> type 'Plain'

template Plain ns::bar<Plain>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:29> col:1 explicit_instantiation_definition 'bar'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'Plain'
// CHECK-NEXT: RecordTypeLoc <col:10> 'Plain'
// CHECK:      TemplateArgument <col:24> type 'Plain'

extern template Color ns::bar<Color>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:36> col:8 explicit_instantiation_declaration extern 'bar'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'Color'
// CHECK-NEXT: EnumTypeLoc <col:17> 'Color'
// CHECK:      TemplateArgument <col:31> type 'Color'

extern template Wrap<int> ns::bar<Wrap<int>>;
// CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:1, col:44> col:8 explicit_instantiation_declaration extern 'bar'
// CHECK-NEXT: NestedNameSpecifier Namespace {{.*}} 'ns'
// CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'Wrap<int>'
// CHECK-NEXT: TemplateSpecializationTypeLoc <col:17, col:25> 'Wrap<int>'

namespace ns {
  extern template Wrap<int> bar<Wrap<int>>;
  // CHECK: ExplicitInstantiationDecl {{.*}} <line:[[@LINE-1]]:3, col:42> col:10 explicit_instantiation_declaration extern 'bar'
  // CHECK-NOT: NestedNameSpecifier
  // CHECK-NEXT: VarTemplateSpecialization {{.*}} 'bar' 'Wrap<int>'
  // CHECK-NEXT: TemplateSpecializationTypeLoc <col:19, col:27> 'Wrap<int>'
}
