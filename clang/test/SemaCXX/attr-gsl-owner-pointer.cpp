// RUN: %clang_cc1 -verify -ast-dump %s | \
// RUN: FileCheck --implicit-check-not OwnerAttr --implicit-check-not PointerAttr %s

int [[gsl::Owner]] i;
// expected-error@-1 {{'Owner' attribute cannot be applied to types}}
void [[gsl::Owner]] f();
// expected-error@-1 {{'Owner' attribute cannot be applied to types}}

[[gsl::Owner]] void f();
// expected-warning@-1 {{'Owner' attribute only applies to structs}}

union [[gsl::Owner(int)]] Union{};
// expected-warning@-1 {{'Owner' attribute only applies to structs}}

struct S {
};

S [[gsl::Owner]] Instance;
// expected-error@-1 {{'Owner' attribute cannot be applied to types}}

class [[gsl::Owner(7)]] OwnerDerefNoType{};
// expected-error@-1 {{expected a type}}

class [[gsl::Pointer("int")]] PointerDerefNoType{};
// expected-error@-1 {{expected a type}}

class [[gsl::Owner(int)]] [[gsl::Pointer(int)]] BothOwnerPointer{};
// expected-error@-1 {{'Pointer' and 'Owner' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}
// CHECK: CXXRecordDecl {{.*}} BothOwnerPointer
// CHECK: OwnerAttr {{.*}} int

class [[gsl::Owner(void)]] OwnerVoidDerefType{};
// expected-error@-1 {{'void' is an invalid argument to attribute 'Owner'}}
class [[gsl::Pointer(void)]] PointerVoidDerefType{};
// expected-error@-1 {{'void' is an invalid argument to attribute 'Pointer'}}

class [[gsl::Pointer(int)]] AddConflictLater{};
// CHECK: CXXRecordDecl {{.*}} AddConflictLater
// CHECK: PointerAttr {{.*}} int
class [[gsl::Owner(int)]] AddConflictLater;
// expected-error@-1 {{'Owner' and 'Pointer' attributes are not compatible}}
// expected-note@-5 {{conflicting attribute is here}}
// CHECK: CXXRecordDecl {{.*}} AddConflictLater
// CHECK: PointerAttr {{.*}} Inherited int

class [[gsl::Owner(int)]] AddConflictLater2{};
// CHECK: CXXRecordDecl {{.*}} AddConflictLater2
// CHECK: OwnerAttr {{.*}} int
class [[gsl::Owner(float)]] AddConflictLater2;
// expected-error@-1 {{'Owner' and 'Owner' attributes are not compatible}}
// expected-note@-5 {{conflicting attribute is here}}
// CHECK: CXXRecordDecl {{.*}} AddConflictLater
// CHECK: OwnerAttr {{.*}} Inherited int

class [[gsl::Owner()]] [[gsl::Owner(int)]] WithAndWithoutParameter{};
// expected-error@-1 {{'Owner' and 'Owner' attributes are not compatible}}
// expected-note@-2 {{conflicting attribute is here}}
// CHECK: CXXRecordDecl {{.*}} WithAndWithoutParameter
// CHECK: OwnerAttr

class [[gsl::Owner(int &)]] ReferenceType{};
// expected-error@-1 {{a reference type is an invalid argument to attribute 'Owner'}}

class [[gsl::Pointer(int[])]] ArrayType{};
// expected-error@-1 {{an array type is an invalid argument to attribute 'Pointer'}}

class [[gsl::Owner]] OwnerMissingParameter{};
// CHECK: CXXRecordDecl {{.*}} OwnerMissingParameter
// CHECK: OwnerAttr

class [[gsl::Pointer]] PointerMissingParameter{};
// CHECK: CXXRecordDecl {{.*}} PointerMissingParameter
// CHECK: PointerAttr

class [[gsl::Owner()]] OwnerWithEmptyParameterList{};
// CHECK: CXXRecordDecl {{.*}} OwnerWithEmptyParameterList
// CHECK: OwnerAttr {{.*}}

class [[gsl::Pointer()]] PointerWithEmptyParameterList{};
// CHECK: CXXRecordDecl {{.*}} PointerWithEmptyParameterList
// CHECK: PointerAttr {{.*}}

struct [[gsl::Owner(int)]] AnOwner{};
// CHECK: CXXRecordDecl {{.*}} AnOwner
// CHECK: OwnerAttr {{.*}} int

struct S;
class [[gsl::Pointer(S)]] APointer{};
// CHECK: CXXRecordDecl {{.*}} APointer
// CHECK: PointerAttr {{.*}} S

class [[gsl::Owner(int)]] [[gsl::Owner(int)]] DuplicateOwner{};
// CHECK: CXXRecordDecl {{.*}} DuplicateOwner
// CHECK: OwnerAttr {{.*}} int

class [[gsl::Pointer(int)]] [[gsl::Pointer(int)]] DuplicatePointer{};
// CHECK: CXXRecordDecl {{.*}} DuplicatePointer
// CHECK: PointerAttr {{.*}} int

class [[gsl::Owner(int)]] AddTheSameLater{};
// CHECK: CXXRecordDecl {{.*}} AddTheSameLater
// CHECK: OwnerAttr {{.*}} int

class [[gsl::Owner(int)]] AddTheSameLater;
// CHECK: CXXRecordDecl {{.*}} prev {{.*}} AddTheSameLater
// CHECK: OwnerAttr {{.*}} int

template <class T>
class [[gsl::Owner]] ForwardDeclared;
// CHECK: ClassTemplateDecl {{.*}} ForwardDeclared
// CHECK: OwnerAttr {{.*}}
// CHECK: ClassTemplateSpecializationDecl {{.*}} ForwardDeclared
// CHECK: TemplateArgument type 'int'
// CHECK: OwnerAttr {{.*}}

template <class T>
class [[gsl::Owner]] ForwardDeclared {
// CHECK: ClassTemplateDecl {{.*}} ForwardDeclared
// CHECK: CXXRecordDecl {{.*}} ForwardDeclared definition
// CHECK: OwnerAttr {{.*}}
};

static_assert(sizeof(ForwardDeclared<int>), ""); // Force instantiation.
