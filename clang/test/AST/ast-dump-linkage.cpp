// RUN: %clang_cc1 -ast-dump -std=c++2c %s | FileCheck --match-full-lines %s

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
// CHECK: |-EnumDecl {{.*}}
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
// FIXME: Why BindingDecls are duplicated?

int Int = 0;
// CHECK: |-VarDecl {{.*}} Int 'int' cinit external-linkage

const int ConstInt = 0;
// CHECK: |-VarDecl {{.*}} ConstInt 'const int' cinit internal-linkage

template <typename T>
T TemplatedVar = T{};
// CHECK: |-VarTemplateDecl {{.*}} TemplatedVar external-linkage
// CHECK: | |-VarDecl {{.*}} TemplatedVar 'T' cinit instantiated_from 0x{{[0-9a-f]*}}
// CHECK: | `-VarTemplateSpecializationDecl {{.*}} used TemplatedVar 'int' implicit_instantiation cinit instantiated_from 0x{{[0-9a-f]*}} external-linkage

// FIXME: VarTemplateSpecializationDecl node is printed twice.

int TemplatedVarSpec = TemplatedVar<int>;
// CHECK: |-VarDecl {{.*}} TemplatedVarSpec 'int' cinit external-linkage
// CHECK: |-VarTemplateSpecializationDecl {{.*}} used TemplatedVar 'int' implicit_instantiation cinit instantiated_from 0x{{[0-9a-f]*}} external-linkage

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
// CHECK: |     | `-VarDecl {{.*}} Int 'int'
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
// CHECK: | `-FunctionDecl {{.*}} TemplatedFuncDef 'void ()'

namespace Known {

void FuncDecl();
// CHECK: | |-FunctionDecl {{.*}} FuncDecl 'void ()' external-linkage

constexpr void ConstexprFuncDecl();
// CHECK: | |-FunctionDecl{{.*}} constexpr ConstexprFuncDecl 'void ()' implicit-inline external-linkage

consteval void ConstevalFuncDecl();
// CHECK: | |-FunctionDecl{{.*}} consteval ConstevalFuncDecl 'void ()' implicit-inline external-linkage
// FIXME: Why consteval functions have linkage?

template <typename>
void TemplatedFuncDecl();
// CHECK: | |-FunctionTemplateDecl {{.*}} TemplatedFuncDecl external-linkage
// CHECK: | | `-FunctionDecl {{.*}} TemplatedFuncDecl 'void ()'

template <typename>
constexpr void TemplatedConstexprFuncDecl();
// CHECK: | |-FunctionTemplateDecl {{.*}} TemplatedConstexprFuncDecl external-linkage
// CHECK: | | `-FunctionDecl {{.*}} constexpr TemplatedConstexprFuncDecl 'void ()' implicit-inline

template <typename>
consteval void TemplatedConstevalFuncDecl();
// CHECK: | |-FunctionTemplateDecl {{.*}} TemplatedConstevalFuncDecl external-linkage
// CHECK: | | `-FunctionDecl {{.*}} consteval TemplatedConstevalFuncDecl 'void ()' implicit-inline
// FIXME: Should consteval function templates have linkage?

struct FriendStruct;
// CHECK: | |-CXXRecordDecl {{.*}} struct FriendStruct external-linkage

template <typename>
struct FriendStructTemplate;
// CHECK: | `-ClassTemplateDecl {{.*}} FriendStructTemplate external-linkage
// CHECK: |   `-CXXRecordDecl {{.*}} struct FriendStructTemplate

} // namespace Known

namespace N {
// CHECK-LABEL: |-NamespaceDecl {{.*}} N external-linkage

struct Struct {
// CHECK: | `-CXXRecordDecl {{.*}} struct Struct definition external-linkage
// CHECK: |   |-CXXRecordDecl {{.*}} implicit struct Struct

  friend struct UnknownFriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct UnknownFriendStruct'
// CHECK: |   | `-CXXRecordDecl {{.*}} friend_undeclared struct UnknownFriendStruct external-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  template <typename>
  friend struct UnknownFriendStructTemplate;
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-ClassTemplateDecl {{.*}} friend_undeclared UnknownFriendStructTemplate external-linkage
// CHECK: |   |   `-CXXRecordDecl {{.*}} struct UnknownFriendStructTemplate
// FIXME: Friend declarations do not bind names, so they cannot have linkage.
  
  friend struct Known::FriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct Known::FriendStruct'
// FIXME: Where is CXXRecordDecl?
  
  template <typename>
  friend struct Known::FriendStructTemplate;
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-ClassTemplateDecl {{.*}} friend FriendStructTemplate external-linkage
// CHECK: |   |   `-CXXRecordDecl {{.*}} struct FriendStructTemplate

  friend void UnknownFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared UnknownFuncDecl 'void ()' external-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  friend constexpr void UnknownConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} constexpr friend_undeclared UnknownConstexprFuncDecl 'void ()' implicit-inline external-linkage

  friend consteval void UnknownConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} consteval friend_undeclared UnknownConstevalFuncDecl 'void ()' implicit-inline external-linkage

  template <typename>
  friend void UnknownFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared UnknownFuncDecl external-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  template <typename>
  friend constexpr void UnknownConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared UnknownConstexprFuncDecl external-linkage

  template <typename>
  friend consteval void UnknownConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared UnknownConstevalFuncDecl external-linkage

  friend void Known::FuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend FuncDecl 'void ()' external-linkage

  friend constexpr void Known::ConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} constexpr friend ConstexprFuncDecl 'void ()' implicit-inline external-linkage

  friend consteval void Known::ConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} consteval friend ConstevalFuncDecl 'void ()' implicit-inline external-linkage

  template <typename>
  friend void Known::TemplatedFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedFuncDecl external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} friend TemplatedFuncDecl 'void ()'

  template <typename>
  friend constexpr void Known::TemplatedConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedConstexprFuncDecl external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} constexpr friend TemplatedConstexprFuncDecl 'void ()' implicit-inline

  template <typename>
  friend consteval void Known::TemplatedConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedConstevalFuncDecl external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} consteval friend TemplatedConstevalFuncDecl 'void ()' implicit-inline

  friend void HiddenFriend() {}
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared HiddenFriend 'void ()' implicit-inline external-linkage

  template <typename>
  friend void HiddenFriendTemplate() {}
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared HiddenFriendTemplate external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} friend_undeclared HiddenFriendTemplate 'void ()' implicit-inline

  int NonStaticDataMember;
// CHECK: |   |-FieldDecl {{.*}} NonStaticDataMember 'int'

  static int StaticDataMember;
// CHECK: |   |-VarDecl {{.*}} StaticDataMember 'int' static external-linkage

  void NonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} NonStaticMemberFunction 'void ()' external-linkage

  constexpr void ConstexprNonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} constexpr ConstexprNonStaticMemberFunction 'void ()' implicit-inline external-linkage

  consteval void ConstevalNonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} consteval ConstevalNonStaticMemberFunction 'void ()' implicit-inline external-linkage

  template <typename>
  void NonStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} NonStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} NonStaticMemberFunctionTemplate 'void ()'

  template <typename>
  constexpr void ConstexprNonStaticMemberFunction();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstexprNonStaticMemberFunction external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} constexpr ConstexprNonStaticMemberFunction 'void ()' implicit-inline

  template <typename>
  consteval void ConstevalNonStaticMemberFunction();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstevalNonStaticMemberFunction external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} consteval ConstevalNonStaticMemberFunction 'void ()' implicit-inline

  static void StaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} StaticMemberFunction 'void ()' static external-linkage

  constexpr static void ConstexprStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} constexpr ConstexprStaticMemberFunction 'void ()' static implicit-inline external-linkage

  consteval static void ConstevalStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} consteval ConstevalStaticMemberFunction 'void ()' static implicit-inline external-linkage

  template <typename>
  static void StaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} StaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} StaticMemberFunctionTemplate 'void ()' static

  template <typename>
  constexpr static void ConstexprStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstexprStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} ConstexprStaticMemberFunctionTemplate 'void ()' static implicit-inline

  template <typename>
  consteval static void ConstevalStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstevalStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} ConstevalStaticMemberFunctionTemplate 'void ()' static implicit-inline

  struct NestedStruct {};
// CHECK: |   |-CXXRecordDecl {{.*}} struct NestedStruct definition external-linkage
// CHECK: |   | `-CXXRecordDecl {{.*}} implicit struct NestedStruct

  template <typename>
  struct NestedStructTemplate {};
// CHECK: |   `-ClassTemplateDecl {{.*}} NestedStructTemplate external-linkage
// CHECK: |     `-CXXRecordDecl {{.*}} struct NestedStructTemplate definition
// CHECK: |       `-CXXRecordDecl {{.*}} implicit struct NestedStructTemplate
};
} // namespace N

namespace M {
// CHECK-LABEL: |-NamespaceDecl {{.*}} M external-linkage

template <typename>
struct StructTemplate {
// CHECK: | `-ClassTemplateDecl {{.*}} StructTemplate external-linkage
// CHECK: | `-CXXRecordDecl {{.*}} struct StructTemplate definition
// CHECK: |   |-CXXRecordDecl {{.*}} implicit struct StructTemplate

  friend struct UnknownFriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct UnknownFriendStruct'
// CHECK: |   | `-CXXRecordDecl {{.*}} friend_undeclared struct UnknownFriendStruct external-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  template <typename>
  friend struct UnknownFriendStructTemplate;
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-ClassTemplateDecl {{.*}} friend_undeclared UnknownFriendStructTemplate external-linkage
// CHECK: |   |   `-CXXRecordDecl {{.*}} struct UnknownFriendStructTemplate
// FIXME: Friend declarations do not bind names, so they cannot have linkage.
  
  friend struct Known::FriendStruct;
// CHECK: |   |-FriendDecl {{.*}} 'struct Known::FriendStruct'
  
  template <typename>
  friend struct Known::FriendStructTemplate;
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-ClassTemplateDecl {{.*}} friend_undeclared FriendStructTemplate external-linkage
// CHECK: |   |   `-CXXRecordDecl {{.*}} struct FriendStructTemplate

  friend void UnknownFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared UnknownFuncDecl 'void ()' external-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  friend constexpr void UnknownConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} constexpr friend_undeclared UnknownConstexprFuncDecl 'void ()' implicit-inline external-linkage

  friend consteval void UnknownConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} consteval friend_undeclared UnknownConstevalFuncDecl 'void ()' implicit-inline external-linkage

  template <typename>
  friend void UnknownFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared UnknownFuncDecl external-linkage
// FIXME: Friend declarations do not bind names, so they cannot have linkage.

  template <typename>
  friend constexpr void UnknownConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared UnknownConstexprFuncDecl external-linkage

  template <typename>
  friend consteval void UnknownConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared UnknownConstevalFuncDecl external-linkage

  friend void Known::FuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared FuncDecl 'void ()' external-linkage

  friend constexpr void Known::ConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} constexpr friend_undeclared ConstexprFuncDecl 'void ()' implicit-inline external-linkage

  friend consteval void Known::ConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} consteval friend_undeclared ConstevalFuncDecl 'void ()' implicit-inline external-linkage

  template <typename>
  friend void Known::TemplatedFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedFuncDecl external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} friend TemplatedFuncDecl 'void ()'

  template <typename>
  friend constexpr void Known::TemplatedConstexprFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedConstexprFuncDecl external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} constexpr friend TemplatedConstexprFuncDecl 'void ()' implicit-inline

  template <typename>
  friend consteval void Known::TemplatedConstevalFuncDecl();
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend TemplatedConstevalFuncDecl external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} consteval friend TemplatedConstevalFuncDecl 'void ()' implicit-inline

  friend void HiddenFriend() {}
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionDecl {{.*}} friend_undeclared HiddenFriend 'void ()' implicit-inline external-linkage

  template <typename>
  friend void HiddenFriendTemplate() {}
// CHECK: |   |-FriendDecl {{.*}} col:{{[0-9]*$}}
// CHECK: |   | `-FunctionTemplateDecl {{.*}} friend_undeclared HiddenFriendTemplate external-linkage
// CHECK: |   |   `-FunctionDecl {{.*}} friend_undeclared HiddenFriendTemplate 'void ()' implicit-inline

  int NonStaticDataMember;
// CHECK: |   |-FieldDecl {{.*}} NonStaticDataMember 'int'

  static int StaticDataMember;
// CHECK: |   |-VarDecl {{.*}} StaticDataMember 'int' static external-linkage

  void NonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} NonStaticMemberFunction 'void ()' external-linkage

  constexpr void ConstexprNonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} constexpr ConstexprNonStaticMemberFunction 'void ()' implicit-inline external-linkage

  consteval void ConstevalNonStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} consteval ConstevalNonStaticMemberFunction 'void ()' implicit-inline external-linkage

  template <typename>
  void NonStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} NonStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} NonStaticMemberFunctionTemplate 'void ()'

  template <typename>
  constexpr void ConstexprNonStaticMemberFunction();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstexprNonStaticMemberFunction external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} constexpr ConstexprNonStaticMemberFunction 'void ()' implicit-inline

  template <typename>
  consteval void ConstevalNonStaticMemberFunction();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstevalNonStaticMemberFunction external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} consteval ConstevalNonStaticMemberFunction 'void ()' implicit-inline

  static void StaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} StaticMemberFunction 'void ()' static external-linkage

  constexpr static void ConstexprStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} constexpr ConstexprStaticMemberFunction 'void ()' static implicit-inline external-linkage

  consteval static void ConstevalStaticMemberFunction();
// CHECK: |   |-CXXMethodDecl {{.*}} consteval ConstevalStaticMemberFunction 'void ()' static implicit-inline external-linkage

  template <typename>
  static void StaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} StaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} StaticMemberFunctionTemplate 'void ()' static

  template <typename>
  constexpr static void ConstexprStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstexprStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} ConstexprStaticMemberFunctionTemplate 'void ()' static implicit-inline

  template <typename>
  consteval static void ConstevalStaticMemberFunctionTemplate();
// CHECK: |   |-FunctionTemplateDecl {{.*}} ConstevalStaticMemberFunctionTemplate external-linkage
// CHECK: |   | `-CXXMethodDecl {{.*}} ConstevalStaticMemberFunctionTemplate 'void ()' static implicit-inline

  struct NestedStruct {};
// CHECK: |   |-CXXRecordDecl {{.*}} struct NestedStruct definition external-linkage
// CHECK: |   | `-CXXRecordDecl {{.*}} implicit struct NestedStruct

  template <typename>
  struct NestedStructTemplate {};
// CHECK: |   `-ClassTemplateDecl {{.*}} NestedStructTemplate external-linkage
// CHECK: |     `-CXXRecordDecl {{.*}} struct NestedStructTemplate definition
// CHECK: |       `-CXXRecordDecl {{.*}} implicit struct NestedStructTemplate
};
} // namespace M

namespace NamespaceAlias = N;
// CHECK: `-NamespaceAliasDecl {{.*}} NamespaceAlias
