// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- -- -I%S/Inputs/use-internal-linkage
// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: {misc-use-internal-linkage.FixMode: 'UseStatic'}}"  -- -I%S/Inputs/use-internal-linkage

#include "tag.h"

struct StructDeclaredInHeader {};
union UnionDeclaredInHeader {};
class ClassDeclaredInHeader {};
enum EnumDeclaredInHeader : int {};
template <typename T> class TemplateDeclaredInHeader<T *> {};
template <> class TemplateDeclaredInHeader<int> {};

struct StructWithNoDefinition;
union UnionWithNoDefinition;
class ClassWithNoDefinition;
enum EnumWithNoDefinition : int;

namespace {

struct StructAlreadyInAnonymousNamespace {};
union UnionAlreadyInAnonymousNamespace {};
class ClassAlreadyInAnonymousNamespace {};
enum EnumAlreadyInAnonymousNamespace : int {};
typedef struct {} TypedefedStructAlreadyInAnonymousNamespace;

} // namespace

struct S {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'S' can be moved into an anonymous namespace to enforce internal linkage
union U {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: union 'U' can be moved into an anonymous namespace to enforce internal linkage
class C {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class 'C' can be moved into an anonymous namespace to enforce internal linkage
enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' can be moved into an anonymous namespace to enforce internal linkage

template <typename>
class Template {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class 'Template' can be moved into an anonymous namespace to enforce internal linkage

struct OuterStruct {
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'OuterStruct' can be moved into an anonymous namespace to enforce internal linkage

  // No warnings for the inner members.
  struct InnerStruct {};
  union InnerUnion {};
  class InnerClass {};
  enum InnerEnum {};
  struct InnerStructDefinedOutOfLine;
};
struct OuterStruct::InnerStructDefinedOutOfLine {};

void f() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'f' can be made static or moved into an anonymous namespace to enforce internal linkage
  struct StructInsideFunction {};
}

namespace ns {

struct S {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'S' can be moved into an anonymous namespace to enforce internal linkage
union U {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: union 'U' can be moved into an anonymous namespace to enforce internal linkage
class C {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: class 'C' can be moved into an anonymous namespace to enforce internal linkage
enum E {};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: enum 'E' can be moved into an anonymous namespace to enforce internal linkage

} // namespace ns

typedef struct {} TypedefedStruct;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: struct 'TypedefedStruct' can be moved into an anonymous namespace to enforce internal linkage

struct Named {} Variable;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'Named' can be moved into an anonymous namespace to enforce internal linkage
// CHECK-MESSAGES: :[[@LINE-2]]:17: warning: variable 'Variable' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static struct Named {} Variable;

struct {} VariableOfUnnamedType;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: variable 'VariableOfUnnamedType' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static struct {} VariableOfUnnamedType;

extern "C" struct MarkedExternC { int i; };

extern "C" {

struct InExternCBlock { int i; };

}
