// RUN: %clang_cc1 -ast-dump %s | FileCheck --match-full-lines %s

namespace std {
template <typename T>
struct initializer_list {
  const T* begin;
  const T* end;
};
} // namespace std

typedef int TypedefInt;
// CHECK: |-TypedefDecl {{.*}} TypedefInt 'int'

using UsingInt = int;
// CHECK: |-TypeAliasDecl {{.*}} UsingInt 'int'

template <typename T>
using TemplateUsingInt = T;
// CHECK: |-TypeAliasTemplateDecl {{.*}} TemplateUsingInt external-linkage
// CHECK: | `-TypeAliasDecl {{.*}} TemplateUsingInt 'T'

typedef struct {} TypedefUnnamedStruct;
// CHECK: |-TypedefDecl {{.*}} TypedefUnnamedStruct 'struct TypedefUnnamedStruct' external-linkage
// CHECK: | `-RecordType {{.*}} 'struct TypedefUnnamedStruct' owns_tag struct

using UsingUnnamedStruct = struct {};
// CHECK: |-TypeAliasDecl {{.*}} UsingUnnamedStruct 'struct UsingUnnamedStruct' external-linkage
// CHECK: | `-RecordType {{.*}} 'struct UsingUnnamedStruct' owns_tag struct

typedef struct TypedefNamedStruct {} TypedefNameForNamedStruct;
// CHECK: |-CXXRecordDecl {{.*}} struct TypedefNamedStruct definition external-linkage
// CHECK: | `-CXXRecordDecl {{.*}} implicit struct TypedefNamedStruct
// CHECK: |-TypedefDecl {{.*}} TypedefNameForNamedStruct 'struct TypedefNamedStruct'
// CHECK: | `-RecordType {{.*}} 'struct TypedefNamedStruct' owns_tag struct
// CHECK: |   `-CXXRecord {{.*}} 'TypedefNamedStruct'

using AliasForNamedStruct = struct UsingNamedStruct {};
// CHECK: |-CXXRecordDecl {{.*}} struct UsingNamedStruct definition external-linkage
// CHECK: | `-CXXRecordDecl {{.*}} implicit struct UsingNamedStruct
// CHECK: |-TypeAliasDecl {{.*}} AliasForNamedStruct 'struct UsingNamedStruct'
// CHECK: | `-RecordType {{.*}} 'struct UsingNamedStruct' owns_tag struct
// CHECK: |   `-CXXRecord {{.*}} 'UsingNamedStruct'

typedef enum {} TypedefUnnamedEnum;
// CHECK: |-TypedefDecl {{.*}} TypedefUnnamedEnum 'enum TypedefUnnamedEnum' external-linkage
// CHECK: | `-EnumType {{.*}} 'enum TypedefUnnamedEnum' owns_tag enum

using UsingUnnamedEnum = enum {};
// CHECK: |-TypeAliasDecl {{.*}} UsingUnnamedEnum 'enum UsingUnnamedEnum' external-linkage
// CHECK: | `-EnumType {{.*}} 'enum UsingUnnamedEnum' owns_tag enum

enum Enum {};
// CHECK: |-EnumDecl {{.*}} Enum external-linkage

enum { Enumerator };
// CHECK: |-EnumDecl {{.*}} no-linkage
// CHECK: | `-EnumConstantDecl {{.*}} referenced Enumerator '(unnamed enum at {{.*}})'
// FIXME: This enum has enumerator as its name for linkage purposes, and has external linkage.

decltype(Enumerator) f();
// CHECK: |-FunctionDecl {{.*}} f 'decltype(Enumerator) ()' external-linkage

auto [Binding1, Binding2] = {3, 4};
// CHECK: |-BindingDecl {{.*}} Binding1 'const int *'
// CHECK: |-BindingDecl {{.*}} Binding2 'const int *'
// CHECK: |-DecompositionDecl {{.*}} used 'std::initializer_list<int>' cinit external-linkage
// CHECK: | |-BindingDecl {{.*}} Binding1 'const int *'
// CHECK: | `-BindingDecl {{.*}} Binding2 'const int *'

int Int = 0;
// CHECK: |-VarDecl {{.*}} Int 'int' cinit external-linkage

const int ConstInt = 0;
// CHECK: |-VarDecl {{.*}} ConstInt 'const int' cinit internal-linkage

template <typename T>
T TemplatedVar = T{};
// CHECK: |-VarTemplateDecl {{.*}} TemplatedVar external-linkage
// CHECK: | |-VarDecl {{.*}} TemplatedVar 'T' cinit instantiated_from 0x{{[0-9a-f]*}} external-linkage
// CHECK: | `-VarTemplateSpecializationDecl {{.*}} used TemplatedVar 'int' implicit_instantiation cinit instantiated_from 0x{{[0-9a-f]*}} external-linkage

int TemplatedVarSpec = TemplatedVar<int>;
// CHECK: |-VarDecl {{.*}} TemplatedVarSpec 'int' cinit external-linkage
// CHECK: |-VarTemplateSpecializationDecl {{.*}} used TemplatedVar 'int' implicit_instantiation cinit instantiated_from 0x{{[0-9a-f]*}} external-linkage

void FuncDecl();
// CHECK: |-FunctionDecl {{.*}} FuncDecl 'void ()' external-linkage

template <typename>
void TemplatedFuncDecl();
// CHECK: |-FunctionTemplateDecl {{.*}} TemplatedFuncDecl external-linkage
// CHECK: | `-FunctionDecl {{.*}} TemplatedFuncDecl 'void ()' external-linkage

void FuncDef() {
// CHECK: |-FunctionDecl {{.*}} FuncDef 'void ()' external-linkage
  
  extern int Int;
// CHECK: |   | `-VarDecl {{.*}} Int 'int' extern external-linkage
  extern const int ConstInt;
// CHECK: |   | `-VarDecl {{.*}} ConstInt 'const int' extern internal-linkage
  void FuncDecl();
// CHECK: |   | `-FunctionDecl {{.*}} FuncDecl 'void ()' external-linkage
  extern void FuncDecl();
// CHECK: |   | `-FunctionDecl {{.*}} FuncDecl 'void ()' extern external-linkage
  {
    int Int;
// CHECK: |     | `-VarDecl {{.*}} Int 'int' no-linkage
    {
      extern int Int;
// CHECK: |       | `-VarDecl {{.*}} Int 'int' extern external-linkage
      extern const int ConstInt;
// CHECK: |         `-VarDecl {{.*}} ConstInt 'const int' extern internal-linkage
    }
  }
}

template <typename>
void TemplatedFuncDef() {}
// CHECK: |-FunctionTemplateDecl {{.*}} TemplatedFuncDef external-linkage
// CHECK: | `-FunctionDecl {{.*}} TemplatedFuncDef 'void ()' external-linkage


struct KnownFriendStruct;
// CHECK: |-CXXRecordDecl {{.*}} struct KnownFriendStruct external-linkage

template <typename>
struct KnownFriendStructTemplate;
// CHECK: |-ClassTemplateDecl {{.*}} KnownFriendStructTemplate external-linkage
// CHECK: | `-CXXRecordDecl {{.*}} struct KnownFriendStructTemplate external-linkage

namespace N {
// CHECK: |-NamespaceDecl {{.*}} N external-linkage

struct Struct {
// CHECK: | `-CXXRecordDecl {{.*}} struct Struct definition external-linkage
// CHECK: |   |-CXXRecordDecl {{.*}} implicit struct Struct

  friend struct UnknownFriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct UnknownFriendStruct'
// CHECK: |   | `-CXXRecordDecl {{.*}} friend_undeclared struct UnknownFriendStruct
  
  friend struct ::KnownFriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct ::KnownFriendStruct'
  
  template <typename>
  friend struct ::KnownFriendStructTemplate;
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-ClassTemplateDecl {{.*}} friend KnownFriendStructTemplate
// CHECK: |   |   `-CXXRecordDecl {{.*}} struct KnownFriendStructTemplate

  friend void UnknownFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared UnknownFuncDecl 'void ()'

  friend void ::FuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend FuncDecl 'void ()'
  
  template <typename>
  friend void ::TemplatedFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedFuncDecl
// CHECK: |   |   `-FunctionDecl {{.*}} friend TemplatedFuncDecl 'void ()'

  int NonStaticDataMember;
// CHECK: |   |-FieldDecl {{.*}} NonStaticDataMember 'int'

  static int StaticDataMember;
// CHECK: |   |-VarDecl {{.*}} StaticDataMember 'int' static external-linkage

  void NonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} NonStaticMemberFunction 'void ()' external-linkage

  template <typename>
  void NonStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} NonStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} NonStaticMemberFunctionTemplate 'void ()' external-linkage

  static void StaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} StaticMemberFunction 'void ()' static external-linkage

  template <typename>
  static void StaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} StaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} StaticMemberFunctionTemplate 'void ()' static external-linkage

  struct NestedStruct {};
// CHECK: |   |-CXXRecordDecl {{.*}} struct NestedStruct definition external-linkage
// CHECK: |   | `-CXXRecordDecl {{.*}} implicit struct NestedStruct

  template <typename>
  struct NestedStructTemplate {};
// CHECK: |   `-ClassTemplateDecl {{.*}} NestedStructTemplate external-linkage
// CHECK: |     `-CXXRecordDecl {{.*}} struct NestedStructTemplate definition external-linkage
// CHECK: |       `-CXXRecordDecl {{.*}} implicit struct NestedStructTemplate
};

} // namespace N

namespace M {
template <typename>
struct StructTemplate {
// CHECK: | `-ClassTemplateDecl {{.*}} StructTemplate external-linkage
// CHECK: |   `-CXXRecordDecl {{.*}} struct StructTemplate definition external-linkage
// CHECK: |     |-CXXRecordDecl {{.*}} implicit struct StructTemplate

  friend struct UnknownFriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct UnknownFriendStruct'
// CHECK: |   | `-CXXRecordDecl {{.*}} friend_undeclared struct UnknownFriendStruct
  
  friend struct ::KnownFriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct ::KnownFriendStruct'
  
  template <typename>
  friend struct ::KnownFriendStructTemplate;
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-ClassTemplateDecl {{.*}} friend_undeclared KnownFriendStructTemplate
// CHECK: |   |   `-CXXRecordDecl {{.*}} struct KnownFriendStructTemplate
  // FIXME: Why "friend_undeclared" instead of "friend"?

  friend void UnknownFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared UnknownFuncDecl 'void ()'

  friend void ::FuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared FuncDecl 'void ()'
  // FIXME: Why "friend_undeclared" instead of "friend"?
  
  template <typename>
  friend void ::TemplatedFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedFuncDecl
// CHECK: |   |   `-FunctionDecl {{.*}} friend TemplatedFuncDecl 'void ()'

  int NonStaticDataMember;
// CHECK: |   |-FieldDecl {{.*}} NonStaticDataMember 'int'

  static int StaticDataMember;
// CHECK: |   |-VarDecl {{.*}} StaticDataMember 'int' static external-linkage

  void NonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} NonStaticMemberFunction 'void ()' external-linkage

  template <typename>
  void NonStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} NonStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} NonStaticMemberFunctionTemplate 'void ()' external-linkage

  static void StaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} StaticMemberFunction 'void ()' static external-linkage

  template <typename>
  static void StaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} StaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} StaticMemberFunctionTemplate 'void ()' static external-linkage

  struct NestedStruct {};
// CHECK: |   |-CXXRecordDecl {{.*}} struct NestedStruct definition external-linkage
// CHECK: |   | `-CXXRecordDecl {{.*}} implicit struct NestedStruct

  template <typename>
  struct NestedStructTemplate {};
// CHECK: |   `-ClassTemplateDecl {{.*}} NestedStructTemplate external-linkage
// CHECK: |     `-CXXRecordDecl {{.*}} struct NestedStructTemplate definition external-linkage
// CHECK: |       `-CXXRecordDecl {{.*}} implicit struct NestedStructTemplate
};
} // namespace M

namespace NamespaceAlias = N;
// CHECK: `-NamespaceAliasDecl {{.*}} NamespaceAlias
