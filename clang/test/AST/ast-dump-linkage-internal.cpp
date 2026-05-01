// RUN: %clang_cc1 -ast-dump %s | FileCheck --match-full-lines %s

namespace std {
template <typename T>
struct initializer_list {
  const T* begin;
  const T* end;
};
} // namespace std

namespace {
// CHECK: |-NamespaceDecl {{.*}} external-linkage
// FIXME: Unnamed namespaces have internal linkage.

typedef int TypedefInt;
// CHECK: | |-TypedefDecl {{.*}} TypedefInt 'int'

using UsingInt = int;
// CHECK: | |-TypeAliasDecl {{.*}} UsingInt 'int'

template <typename T>
using TemplateUsingInt = T;
// CHECK: | |-TypeAliasTemplateDecl {{.*}} TemplateUsingInt internal-linkage
// CHECK: | | `-TypeAliasDecl {{.*}} TemplateUsingInt 'T'

typedef struct {} TypedefUnnamedStruct;
// CHECK: | |-TypedefDecl {{.*}} TypedefUnnamedStruct 'struct TypedefUnnamedStruct' internal-linkage
// CHECK: | | `-RecordType {{.*}} 'struct TypedefUnnamedStruct' owns_tag struct

using UsingUnnamedStruct = struct {};
// CHECK: | |-TypeAliasDecl {{.*}} UsingUnnamedStruct 'struct UsingUnnamedStruct' internal-linkage
// CHECK: | | `-RecordType {{.*}} 'struct UsingUnnamedStruct' owns_tag struct

typedef struct TypedefNamedStruct {} TypedefNameForNamedStruct;
// CHECK: | |-CXXRecordDecl {{.*}} struct TypedefNamedStruct definition internal-linkage
// CHECK: | | `-CXXRecordDecl {{.*}} implicit struct TypedefNamedStruct
// CHECK: | |-TypedefDecl {{.*}} TypedefNameForNamedStruct 'struct TypedefNamedStruct'
// CHECK: | | `-RecordType {{.*}} 'struct TypedefNamedStruct' owns_tag struct
// CHECK: | |   `-CXXRecord {{.*}} 'TypedefNamedStruct'

using AliasForNamedStruct = struct UsingNamedStruct {};
// CHECK: | |-CXXRecordDecl {{.*}} struct UsingNamedStruct definition internal-linkage
// CHECK: | | `-CXXRecordDecl {{.*}} implicit struct UsingNamedStruct
// CHECK: | |-TypeAliasDecl {{.*}} AliasForNamedStruct 'struct UsingNamedStruct'
// CHECK: | | `-RecordType {{.*}} 'struct UsingNamedStruct' owns_tag struct
// CHECK: | |   `-CXXRecord {{.*}} 'UsingNamedStruct'

typedef enum {} TypedefUnnamedEnum;
// CHECK: | |-TypedefDecl {{.*}} TypedefUnnamedEnum 'enum TypedefUnnamedEnum' internal-linkage
// CHECK: | | `-EnumType {{.*}} 'enum TypedefUnnamedEnum' owns_tag enum

using UsingUnnamedEnum = enum {};
// CHECK: | |-TypeAliasDecl {{.*}} UsingUnnamedEnum 'enum UsingUnnamedEnum' internal-linkage
// CHECK: | | `-EnumType {{.*}} 'enum UsingUnnamedEnum' owns_tag enum

enum Enum {};
// CHECK: | |-EnumDecl {{.*}} Enum internal-linkage

enum { Enumerator };
// CHECK: | |-EnumDecl {{.*}} internal-linkage
// CHECK: | | `-EnumConstantDecl {{.*}} referenced Enumerator '(anonymous namespace)::(unnamed enum at {{.*}})'

decltype(Enumerator) f();
// CHECK: | |-FunctionDecl {{.*}} f 'decltype(Enumerator) ()' internal-linkage

auto [Binding1, Binding2] = {3, 4};
// CHECK: | |-BindingDecl {{.*}} Binding1 'const int *'
// CHECK: | |-BindingDecl {{.*}} Binding2 'const int *'
// CHECK: | |-DecompositionDecl {{.*}} used 'std::initializer_list<int>' cinit internal-linkage
// CHECK: | | |-BindingDecl {{.*}} Binding1 'const int *'
// CHECK: | | `-BindingDecl {{.*}} Binding2 'const int *'
// FIXME: Why BindingDecls are duplicated?

int Int = 0;
// CHECK: | |-VarDecl {{.*}} Int 'int' cinit internal-linkage

const int ConstInt = 0;
// CHECK: | |-VarDecl {{.*}} ConstInt 'const int' cinit internal-linkage

template <typename T>
T TemplatedVar = T{};
// CHECK: | |-VarTemplateDecl {{.*}} TemplatedVar internal-linkage
// CHECK: | | |-VarDecl {{.*}} TemplatedVar 'T' cinit instantiated_from 0x{{[0-9a-f]*}}
// CHECK: | | `-VarTemplateSpecializationDecl {{.*}} used TemplatedVar 'int' implicit_instantiation cinit instantiated_from 0x{{[0-9a-f]*}} internal-linkage

// FIXME: VarTemplateSpecializationDecl node is printed twice.

int TemplatedVarSpec = TemplatedVar<int>;
// CHECK: | |-VarDecl {{.*}} TemplatedVarSpec 'int' cinit internal-linkage
// CHECK: | |-VarTemplateSpecializationDecl {{.*}} used TemplatedVar 'int' implicit_instantiation cinit instantiated_from 0x{{[0-9a-f]*}} internal-linkage

void FuncDecl();
// CHECK: | |-FunctionDecl {{.*}} FuncDecl 'void ()' internal-linkage

template <typename>
void TemplatedFuncDecl();
// CHECK: | |-FunctionTemplateDecl {{.*}} TemplatedFuncDecl internal-linkage
// CHECK: | | `-FunctionDecl {{.*}} TemplatedFuncDecl 'void ()'

void FuncDef() {
// CHECK: | |-FunctionDecl {{.*}} FuncDef 'void ()' internal-linkage
  
  extern int Int;
// CHECK: | |   | `-VarDecl {{.*}} Int 'int' extern internal-linkage
  extern const int ConstInt;
// CHECK: | |   | `-VarDecl {{.*}} ConstInt 'const int' extern internal-linkage
  void FuncDecl();
// CHECK: | |   | `-FunctionDecl {{.*}} FuncDecl 'void ()' internal-linkage
  extern void FuncDecl();
// CHECK: | |   | `-FunctionDecl {{.*}} FuncDecl 'void ()' extern internal-linkage
  {
    int Int;
// CHECK: | |     | `-VarDecl {{.*}} Int 'int'
    {
      extern int Int;
// CHECK: | |       | `-VarDecl {{.*}} Int 'int' extern internal-linkage
      extern const int ConstInt;
// CHECK: | |         `-VarDecl {{.*}} ConstInt 'const int' extern internal-linkage
    }
  }
}

template <typename>
void TemplatedFuncDef() {}
// CHECK: | |-FunctionTemplateDecl {{.*}} TemplatedFuncDef internal-linkage
// CHECK: | | `-FunctionDecl {{.*}} TemplatedFuncDef 'void ()'


struct KnownFriendStruct;
// CHECK: | |-CXXRecordDecl {{.*}} struct KnownFriendStruct internal-linkage

template <typename>
struct KnownFriendStructTemplate;
// CHECK: | |-ClassTemplateDecl {{.*}} KnownFriendStructTemplate internal-linkage
// CHECK: | | `-CXXRecordDecl {{.*}} struct KnownFriendStructTemplate

namespace N {
// CHECK: | |-NamespaceDecl {{.*}} N internal-linkage

struct Struct {
// CHECK: | | `-CXXRecordDecl {{.*}} struct Struct definition internal-linkage
// CHECK: | |   |-CXXRecordDecl {{.*}} implicit struct Struct

  friend struct UnknownFriendStruct;
// CHECK: | |   |-FriendDecl {{.*}} 'struct UnknownFriendStruct'
// CHECK: | |   | `-CXXRecordDecl {{.*}} friend_undeclared struct UnknownFriendStruct internal-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  friend struct ::KnownFriendStruct;
// CHECK: | |   |-FriendDecl {{.*}} 'struct ::KnownFriendStruct'
  
  template <typename>
  friend struct KnownFriendStructTemplate;
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-ClassTemplateDecl {{.*}} friend_undeclared KnownFriendStructTemplate internal-linkage
// CHECK: | |   |   `-CXXRecordDecl {{.*}} struct KnownFriendStructTemplate

  friend void UnknownFuncDecl();
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionDecl {{.*}} friend_undeclared UnknownFuncDecl 'void ()' internal-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  friend void FuncDecl();
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionDecl {{.*}} friend_undeclared FuncDecl 'void ()' internal-linkage
  
  template <typename>
  friend void TemplatedFuncDecl();
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionTemplateDecl {{.*}} friend_undeclared TemplatedFuncDecl internal-linkage
// CHECK: | |   |   `-FunctionDecl {{.*}} friend_undeclared TemplatedFuncDecl 'void ()'

  friend void HiddenFriend() {}
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionDecl {{.*}} friend_undeclared HiddenFriend 'void ()' implicit-inline internal-linkage

  template <typename>
  friend void HiddenFriendTemplate() {}
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionTemplateDecl {{.*}} friend_undeclared HiddenFriendTemplate internal-linkage
// CHECK: | |   |   `-FunctionDecl {{.*}} friend_undeclared HiddenFriendTemplate 'void ()' implicit-inline

  int NonStaticDataMember;
// CHECK: | |   |-FieldDecl {{.*}} NonStaticDataMember 'int'

  static int StaticDataMember;
// CHECK: | |   |-VarDecl {{.*}} StaticDataMember 'int' static internal-linkage

  void NonStaticMemberFunction();
// CHECK: | |   |-CXXMethodDecl {{.*}} NonStaticMemberFunction 'void ()' internal-linkage

  template <typename>
  void NonStaticMemberFunctionTemplate();
// CHECK: | |   |-FunctionTemplateDecl {{.*}} NonStaticMemberFunctionTemplate internal-linkage
// CHECK: | |   | `-CXXMethodDecl {{.*}} NonStaticMemberFunctionTemplate 'void ()'

  static void StaticMemberFunction();
// CHECK: | |   |-CXXMethodDecl {{.*}} StaticMemberFunction 'void ()' static internal-linkage

  template <typename>
  static void StaticMemberFunctionTemplate();
// CHECK: | |   |-FunctionTemplateDecl {{.*}} StaticMemberFunctionTemplate internal-linkage
// CHECK: | |   | `-CXXMethodDecl {{.*}} StaticMemberFunctionTemplate 'void ()' static

  struct NestedStruct {};
// CHECK: | |   |-CXXRecordDecl {{.*}} struct NestedStruct definition internal-linkage
// CHECK: | |   | `-CXXRecordDecl {{.*}} implicit struct NestedStruct

  template <typename>
  struct NestedStructTemplate {};
// CHECK: | |   `-ClassTemplateDecl {{.*}} NestedStructTemplate internal-linkage
// CHECK: | |     `-CXXRecordDecl {{.*}} struct NestedStructTemplate definition
// CHECK: | |       `-CXXRecordDecl {{.*}} implicit struct NestedStructTemplate
};

} // namespace N

namespace M {
template <typename>
struct StructTemplate {
// CHECK: | | `-ClassTemplateDecl {{.*}} StructTemplate internal-linkage
// CHECK: | |   `-CXXRecordDecl {{.*}} struct StructTemplate definition
// CHECK: | |     |-CXXRecordDecl {{.*}} implicit struct StructTemplate

  friend struct UnknownFriendStruct;
// CHECK: | |   |-FriendDecl {{.*}} 'struct UnknownFriendStruct'
// CHECK: | |   | `-CXXRecordDecl {{.*}} friend_undeclared struct UnknownFriendStruct internal-linkage
  
  friend struct ::KnownFriendStruct;
// CHECK: | |   |-FriendDecl {{.*}} 'struct ::KnownFriendStruct'
// FIXME: Where is CXXRecordDecl? 
  
  template <typename>
  friend struct KnownFriendStructTemplate;
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-ClassTemplateDecl {{.*}} friend_undeclared KnownFriendStructTemplate internal-linkage
// CHECK: | |   |   `-CXXRecordDecl {{.*}} struct KnownFriendStructTemplate
// FIXME: Why "friend_undeclared" instead of "friend"?

  friend void UnknownFuncDecl();
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionDecl {{.*}} friend_undeclared UnknownFuncDecl 'void ()' internal-linkage

  friend void FuncDecl();
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionDecl {{.*}} friend_undeclared FuncDecl 'void ()' internal-linkage
// FIXME: Why "friend_undeclared" instead of "friend"?
  
  template <typename>
  friend void TemplatedFuncDecl();
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionTemplateDecl {{.*}} friend_undeclared TemplatedFuncDecl internal-linkage
// CHECK: | |   |   `-FunctionDecl {{.*}} friend_undeclared TemplatedFuncDecl 'void ()'

  friend void HiddenFriend() {}
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionDecl {{.*}} friend_undeclared HiddenFriend 'void ()' implicit-inline internal-linkage

  template <typename>
  friend void HiddenFriendTemplate() {}
// CHECK: | |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: | |   | `-FunctionTemplateDecl {{.*}} friend_undeclared HiddenFriendTemplate internal-linkage
// CHECK: | |   |   `-FunctionDecl {{.*}} friend_undeclared HiddenFriendTemplate 'void ()' implicit-inline

  int NonStaticDataMember;
// CHECK: | |   |-FieldDecl {{.*}} NonStaticDataMember 'int'

  static int StaticDataMember;
// CHECK: | |   |-VarDecl {{.*}} StaticDataMember 'int' static internal-linkage

  void NonStaticMemberFunction();
// CHECK: | |   |-CXXMethodDecl {{.*}} NonStaticMemberFunction 'void ()' internal-linkage

  template <typename>
  void NonStaticMemberFunctionTemplate();
// CHECK: | |   |-FunctionTemplateDecl {{.*}} NonStaticMemberFunctionTemplate internal-linkage
// CHECK: | |   | `-CXXMethodDecl {{.*}} NonStaticMemberFunctionTemplate 'void ()'

  static void StaticMemberFunction();
// CHECK: | |   |-CXXMethodDecl {{.*}} StaticMemberFunction 'void ()' static internal-linkage

  template <typename>
  static void StaticMemberFunctionTemplate();
// CHECK: | |   |-FunctionTemplateDecl {{.*}} StaticMemberFunctionTemplate internal-linkage
// CHECK: | |   | `-CXXMethodDecl {{.*}} StaticMemberFunctionTemplate 'void ()' static

  struct NestedStruct {};
// CHECK: | |   |-CXXRecordDecl {{.*}} struct NestedStruct definition internal-linkage
// CHECK: | |   | `-CXXRecordDecl {{.*}} implicit struct NestedStruct

  template <typename>
  struct NestedStructTemplate {};
// CHECK: | |   `-ClassTemplateDecl {{.*}} NestedStructTemplate internal-linkage
// CHECK: | |     `-CXXRecordDecl {{.*}} struct NestedStructTemplate definition
// CHECK: | |       `-CXXRecordDecl {{.*}} implicit struct NestedStructTemplate
};
} // namespace M

namespace NamespaceAlias = N;
// CHECK: | `-NamespaceAliasDecl {{.*}} NamespaceAlias

} // unnamed namespace
