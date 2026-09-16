// RUN: %check_clang_tidy -std=c++11-or-later \
// RUN:   -check-header %S/Inputs/use-default-member-init/non-visible-references.h \
// RUN:   %s modernize-use-default-member-init %t -- -- -I%S/Inputs/use-default-member-init
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffix=ALLOW \
// RUN:   -check-header %S/Inputs/use-default-member-init/non-visible-references.h \
// RUN:   %s modernize-use-default-member-init %t.allow -- \
// RUN:   -config="{CheckOptions: {modernize-use-default-member-init.IgnoreNonVisibleReferences: false}}" -- \
// RUN:   -I%S/Inputs/use-default-member-init

#include "non-visible-references.h"

struct MainPositive {
  MainPositive() : member(42) {}
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member' [modernize-use-default-member-init]
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{42};
  // CHECK-FIXES-ALLOW: int member{42};
};

namespace {
constexpr double CppConstant = 2.0;
static int CppStatic = 3;
enum { CppEnum = 4 };
} // namespace

#define CPP_MACRO_CONSTANT CppConstant

NonVisibleConstexpr::NonVisibleConstexpr() : member(CppConstant) {}
NonVisibleStatic::NonVisibleStatic() : member(CppStatic) {}
NonVisibleEnum::NonVisibleEnum() : member(CppEnum) {}
NonVisibleNestedCast::NonVisibleNestedCast()
    : member(static_cast<int>(CppConstant + 1.0)) {}
NonVisibleMacroReference::NonVisibleMacroReference()
    : member(NON_VISIBLE_INIT_VALUE) {}
using HeaderValues::Constant;
NonVisibleUsingDeclaration::NonVisibleUsingDeclaration() : member(Constant) {}
template <typename T>
NonVisibleTemplate<T>::NonVisibleTemplate() : member(CppConstant) {}
NonVisibleTemplate<int> NonVisibleTemplateInstance;

HeaderVisibleConstexpr::HeaderVisibleConstexpr()
    : member(HeaderValues::Constant) {}
HeaderVisibleStatic::HeaderVisibleStatic() : member(HeaderValues::Static) {}
HeaderVisibleEnum::HeaderVisibleEnum() : member(HeaderValues::EnumValue) {}
SameClassLaterStatic::SameClassLaterStatic() : member(ClassConstant) {}
SameClassLaterEnum::SameClassLaterEnum() : member(ClassEnumValue) {}
InheritedStatic::InheritedStatic() : member(BaseConstant) {}
