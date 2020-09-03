// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -std=c++17 -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck -strict-whitespace %s

void f() {
  auto IsNotGenericLambda = [](){};
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <col:29> col:29 implicit class definition
  // CHECK-NOT: DefinitionData {{.*}}generic{{.*}}
  // CHECK-NEXT: DefinitionData {{.*}}lambda{{.*}}
  auto IsGenericLambda = [](auto){};
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <col:26> col:26 implicit class definition
  // CHECK-NEXT: DefinitionData {{.*}}generic{{.*}}lambda{{.*}}
}

struct CanPassInRegisters {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CanPassInRegisters definition
  // CHECK-NEXT: DefinitionData {{.*}}pass_in_registers{{.*}}
  CanPassInRegisters(const CanPassInRegisters&) = default;
};

struct CantPassInRegisters {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CantPassInRegisters definition
  // CHECK-NOT: DefinitionData {{.*}}pass_in_registers{{.*}}
  CantPassInRegisters(const CantPassInRegisters&) = delete;
};

struct IsEmpty {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct IsEmpty definition
  // CHECK-NEXT: DefinitionData {{.*}}empty{{.*}}
};

struct IsNotEmpty {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotEmpty definition
  // CHECK-NOT: DefinitionData {{.*}}empty{{.*}}
  int a;
};

struct IsAggregate {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsAggregate definition
  // CHECK-NEXT: DefinitionData {{.*}}aggregate{{.*}}
  int a;
};

struct IsNotAggregate {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+4]]:1> line:[[@LINE-1]]:8 struct IsNotAggregate definition
  // CHECK-NOT: DefinitionData {{.*}}aggregate{{.*}}
private:
  int a;
};

struct IsStandardLayout {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsStandardLayout definition
  // CHECK-NEXT: DefinitionData {{.*}}standard_layout{{.*}}
  void f();
};

struct IsNotStandardLayout {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotStandardLayout definition
  // CHECK-NOT: DefinitionData {{.*}}standard_layout{{.*}}
  virtual void f();
};

struct IsTriviallyCopyable {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct IsTriviallyCopyable definition
  // CHECK-NEXT: DefinitionData {{.*}}trivially_copyable{{.*}}
};

struct IsNotTriviallyCopyable {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotTriviallyCopyable definition
  // CHECK-NOT: DefinitionData {{.*}}trivially_copyable{{.*}}
  IsNotTriviallyCopyable(const IsNotTriviallyCopyable&) {}
};

struct IsPOD {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsPOD definition
  // CHECK-NEXT: DefinitionData {{.*}}pod{{.*}}
  int a;
};

struct IsNotPOD {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotPOD definition
  // CHECK-NOT: DefinitionData {{.*}}pod{{.*}}
  int &a;
};

struct IsTrivial {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsTrivial definition
  // CHECK-NEXT: DefinitionData {{.*}}trivial {{.*}}
  IsTrivial() = default;
};

struct IsNotTrivial {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotTrivial definition
  // CHECK-NOT: DefinitionData {{.*}}trivial {{.*}}
  IsNotTrivial() {}
};

struct IsPolymorphic {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsPolymorphic definition
  // CHECK-NEXT: DefinitionData {{.*}}polymorphic{{.*}}
  virtual void f();
};

struct IsNotPolymorphic {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotPolymorphic definition
  // CHECK-NOT: DefinitionData {{.*}}polymorphic{{.*}}
  void f();
};

struct IsAbstract {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsAbstract definition
  // CHECK-NEXT: DefinitionData {{.*}}abstract{{.*}}
  virtual void f() = 0;
};

struct IsNotAbstract {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotAbstract definition
  // CHECK-NOT: DefinitionData {{.*}}abstract{{.*}}
  virtual void f();
};

struct IsLiteral {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsLiteral definition
  // CHECK-NEXT: DefinitionData {{.*}}literal{{.*}}
  ~IsLiteral() = default;
};

struct IsNotLiteral {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct IsNotLiteral definition
  // CHECK-NOT: DefinitionData {{.*}}literal{{.*}}
  ~IsNotLiteral() {}
};

struct HasUserDeclaredConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct HasUserDeclaredConstructor definition
  // CHECK-NEXT: DefinitionData {{.*}}has_user_declared_ctor{{.*}}
  HasUserDeclaredConstructor() {}
};

struct HasNoUserDeclaredConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct HasNoUserDeclaredConstructor definition
  // CHECK-NOT: DefinitionData {{.*}}has_user_declared_ctor{{.*}}
};

struct HasConstexprNonCopyMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct HasConstexprNonCopyMoveConstructor definition
  // CHECK-NEXT: DefinitionData {{.*}}has_constexpr_non_copy_move_ctor{{.*}}
  constexpr HasConstexprNonCopyMoveConstructor() {}
};

struct HasNoConstexprNonCopyMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct HasNoConstexprNonCopyMoveConstructor definition
  // CHECK-NOT: DefinitionData {{.*}}has_constexpr_non_copy_move_ctor{{.*}}
  HasNoConstexprNonCopyMoveConstructor() {}
};

struct HasMutableFields {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct HasMutableFields definition
  // CHECK-NEXT: DefinitionData {{.*}}has_mutable_fields{{.*}}
  mutable int i;
};

struct HasNoMutableFields {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct HasNoMutableFields definition
  // CHECK-NOT: DefinitionData {{.*}}has_mutable_fields{{.*}}
  int i;
};

struct HasVariantMembers {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+5]]:1> line:[[@LINE-1]]:8 struct HasVariantMembers definition
  // CHECK-NEXT: DefinitionData {{.*}}has_variant_members{{.*}}
  union {
    int i;
  };
};

struct HasNoVariantMembers {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct HasNoVariantMembers definition
  // CHECK-NOT: DefinitionData {{.*}}has_variant_members{{.*}}
};

struct AllowsConstDefaultInit {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct AllowsConstDefaultInit definition
  // CHECK-NEXT: DefinitionData {{.*}}can_const_default_init{{.*}}
  int i = 12;
};

struct DoesNotAllowConstDefaultInit {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotAllowConstDefaultInit definition
  // CHECK-NOT: DefinitionData {{.*}}can_const_default_init{{.*}}
  int i;
};
