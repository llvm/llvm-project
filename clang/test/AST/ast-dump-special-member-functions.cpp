// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump %s | FileCheck -strict-whitespace %s

// FIXME: exists

struct TrivialDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <{{.*}}:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct TrivialDefaultConstructor definition
  // CHECK: DefaultConstructor {{.*}} trivial{{.*}}
  TrivialDefaultConstructor() = default;
};

struct NontrivialDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NontrivialDefaultConstructor definition
  // CHECK: DefaultConstructor {{.*}}non_trivial{{.*}}
  NontrivialDefaultConstructor() {}
};

struct UserProvidedDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct UserProvidedDefaultConstructor definition
  // CHECK: DefaultConstructor {{.*}}user_provided{{.*}}
  UserProvidedDefaultConstructor() {}
};

struct NonUserProvidedDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NonUserProvidedDefaultConstructor definition
  // CHECK-NOT: DefaultConstructor {{.*}}user_provided{{.*}}
};

struct HasConstexprDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct HasConstexprDefaultConstructor definition
  // CHECK: DefaultConstructor {{.*}}constexpr{{.*}}
  constexpr HasConstexprDefaultConstructor() {}
};

struct DoesNotHaveConstexprDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotHaveConstexprDefaultConstructor definition
  // CHECK-NOT: DefaultConstructor {{.*}} constexpr{{.*}}
  DoesNotHaveConstexprDefaultConstructor() {}
};

struct NeedsImplicitDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NeedsImplicitDefaultConstructor definition
  // CHECK: DefaultConstructor {{.*}}needs_implicit{{.*}}
  int i = 12;
};

struct DoesNotNeedImplicitDefaultConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotNeedImplicitDefaultConstructor definition
  // CHECK-NOT: DefaultConstructor {{.*}}needs_implicit{{.*}}
  DoesNotNeedImplicitDefaultConstructor() {}
};

struct DefaultedDefaultConstructorIsConstexpr {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DefaultedDefaultConstructorIsConstexpr definition
  // CHECK: DefaultConstructor {{.*}}defaulted_is_constexpr{{.*}}
  DefaultedDefaultConstructorIsConstexpr() = default;
};

struct DefaultedDefaultConstructorIsNotConstexpr {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+6]]:1> line:[[@LINE-1]]:8 struct DefaultedDefaultConstructorIsNotConstexpr definition
  // CHECK-NOT: DefaultConstructor {{.*}}defaulted_is_constexpr{{.*}}
  DefaultedDefaultConstructorIsNotConstexpr() = default;
  union {
    int i;
  };
};

struct SimpleCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct SimpleCopyConstructor definition
  // CHECK: CopyConstructor {{.*}}simple{{.*}}
  int i = 12;
};

struct NotSimpleCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NotSimpleCopyConstructor definition
  // CHECK-NOT: CopyConstructor {{.*}}simple{{.*}}
  NotSimpleCopyConstructor(const NotSimpleCopyConstructor&) = delete;
};

struct TrivialCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct TrivialCopyConstructor definition
  // CHECK: CopyConstructor {{.*}} trivial{{.*}}
  TrivialCopyConstructor() = default;
};

struct NontrivialCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NontrivialCopyConstructor definition
  // CHECK: CopyConstructor {{.*}}non_trivial{{.*}}
  NontrivialCopyConstructor(const NontrivialCopyConstructor&) {}
};

struct UserDeclaredCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct UserDeclaredCopyConstructor definition
  // CHECK: CopyConstructor {{.*}}user_declared{{.*}}
  UserDeclaredCopyConstructor(const UserDeclaredCopyConstructor&) {}
};

struct NonUserDeclaredCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NonUserDeclaredCopyConstructor definition
  // CHECK-NOT: CopyConstructor {{.*}}user_declared{{.*}}
};

struct CopyConstructorHasConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CopyConstructorHasConstParam definition
  // CHECK: CopyConstructor {{.*}}has_const_param{{.*}}
  CopyConstructorHasConstParam(const CopyConstructorHasConstParam&) {}
};

struct CopyConstructorDoesNotHaveConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CopyConstructorDoesNotHaveConstParam definition
  // CHECK-NOT: CopyConstructor {{.*}} has_const_param{{.*}}
  CopyConstructorDoesNotHaveConstParam(CopyConstructorDoesNotHaveConstParam&) {}
};

struct NeedsImplicitCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NeedsImplicitCopyConstructor definition
  // CHECK: CopyConstructor {{.*}}needs_implicit{{.*}}
  int i = 12;
};

struct DoesNotNeedImplicitCopyConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotNeedImplicitCopyConstructor definition
  // CHECK-NOT: CopyConstructor {{.*}}needs_implicit{{.*}}
  DoesNotNeedImplicitCopyConstructor(const DoesNotNeedImplicitCopyConstructor&) {}
};

struct DeletedDestructor {
private:
  ~DeletedDestructor() = delete;
};

struct CopyConstructorNeedsOverloadResolution : virtual DeletedDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct CopyConstructorNeedsOverloadResolution definition
  // CHECK: CopyConstructor {{.*}}needs_overload_resolution{{.*}}
};

struct CopyConstructorDoesNotNeedOverloadResolution {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct CopyConstructorDoesNotNeedOverloadResolution definition
  // CHECK-NOT: CopyConstructor {{.*}}needs_overload_resolution{{.*}}
};

struct DefaultedCopyConstructorIsDeleted {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+4]]:1> line:[[@LINE-1]]:8 struct DefaultedCopyConstructorIsDeleted definition
  // CHECK: CopyConstructor {{.*}}defaulted_is_deleted{{.*}}
  int &&i;
  DefaultedCopyConstructorIsDeleted(const DefaultedCopyConstructorIsDeleted&) = default;
};

struct DefaultedCopyConstructorIsNotDeleted {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+4]]:1> line:[[@LINE-1]]:8 struct DefaultedCopyConstructorIsNotDeleted definition
  // CHECK-NOT: CopyConstructor {{.*}}defaulted_is_deleted{{.*}}
  int i;
  DefaultedCopyConstructorIsNotDeleted(const DefaultedCopyConstructorIsNotDeleted&) = default;
};

struct BaseWithoutCopyConstructorConstParam {
  BaseWithoutCopyConstructorConstParam(BaseWithoutCopyConstructorConstParam&);
};

struct ImplicitCopyConstructorHasConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct ImplicitCopyConstructorHasConstParam definition
  // CHECK: CopyConstructor {{.*}}implicit_has_const_param{{.*}}
};

struct ImplicitCopyConstructorDoesNotHaveConstParam : BaseWithoutCopyConstructorConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct ImplicitCopyConstructorDoesNotHaveConstParam definition
  // CHECK-NOT: CopyConstructor {{.*}}implicit_has_const_param{{.*}}
};

struct MoveConstructorExists {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct MoveConstructorExists definition
  // CHECK: MoveConstructor {{.*}}exists{{.*}}
};

struct MoveConstructorDoesNotExist {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct MoveConstructorDoesNotExist definition
  // CHECK-NOT: MoveConstructor {{.*}}exists{{.*}}
  MoveConstructorDoesNotExist(const MoveConstructorDoesNotExist&);
};

struct SimpleMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct SimpleMoveConstructor definition
  // CHECK: MoveConstructor {{.*}}simple{{.*}}
  int i = 12;
};

struct NotSimpleMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NotSimpleMoveConstructor definition
  // CHECK-NOT: MoveConstructor {{.*}}simple{{.*}}
  NotSimpleMoveConstructor(NotSimpleMoveConstructor&&) = delete;
};

struct TrivialMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct TrivialMoveConstructor definition
  // CHECK: MoveConstructor {{.*}} trivial{{.*}}
  TrivialMoveConstructor() = default;
};

struct NontrivialMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NontrivialMoveConstructor definition
  // CHECK: MoveConstructor {{.*}}non_trivial{{.*}}
  NontrivialMoveConstructor(NontrivialMoveConstructor&&) {}
};

struct UserDeclaredMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct UserDeclaredMoveConstructor definition
  // CHECK: MoveConstructor {{.*}}user_declared{{.*}}
  UserDeclaredMoveConstructor(UserDeclaredMoveConstructor&&) {}
};

struct NonUserDeclaredMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NonUserDeclaredMoveConstructor definition
  // CHECK-NOT: MoveConstructor {{.*}}user_declared{{.*}}
};

struct NeedsImplicitMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NeedsImplicitMoveConstructor definition
  // CHECK: MoveConstructor {{.*}}needs_implicit{{.*}}
  int i = 12;
};

struct DoesNotNeedImplicitMoveConstructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotNeedImplicitMoveConstructor definition
  // CHECK-NOT: MoveConstructor {{.*}}needs_implicit{{.*}}
  DoesNotNeedImplicitMoveConstructor(DoesNotNeedImplicitMoveConstructor&&) {}
};

struct MoveConstructorNeedsOverloadResolution : virtual DeletedDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct MoveConstructorNeedsOverloadResolution definition
  // CHECK: MoveConstructor {{.*}}needs_overload_resolution{{.*}}
};

struct MoveConstructorDoesNotNeedOverloadResolution {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct MoveConstructorDoesNotNeedOverloadResolution definition
  // CHECK-NOT: MoveConstructor {{.*}}needs_overload_resolution{{.*}}
};

// FIXME: defaulted_is_deleted

struct TrivialCopyAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct TrivialCopyAssignment definition
  // CHECK: CopyAssignment {{.*}} trivial{{.*}}
  TrivialCopyAssignment& operator=(const TrivialCopyAssignment&) = default;
};

struct NontrivialCopyAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NontrivialCopyAssignment definition
  // CHECK: CopyAssignment {{.*}}non_trivial{{.*}}
  NontrivialCopyAssignment& operator=(const NontrivialCopyAssignment&) {}
};

struct CopyAssignmentHasConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CopyAssignmentHasConstParam definition
  // CHECK: CopyAssignment {{.*}}has_const_param{{.*}}
  CopyAssignmentHasConstParam& operator=(const CopyAssignmentHasConstParam&) {}
};

struct CopyAssignmentDoesNotHaveConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CopyAssignmentDoesNotHaveConstParam definition
  // CHECK-NOT: CopyAssignment {{.*}} has_const_param{{.*}}
  CopyAssignmentDoesNotHaveConstParam& operator=(CopyAssignmentDoesNotHaveConstParam&) {}
};

struct UserDeclaredCopyAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct UserDeclaredCopyAssignment definition
  // CHECK: CopyAssignment {{.*}}user_declared{{.*}}
  UserDeclaredCopyAssignment& operator=(const UserDeclaredCopyAssignment&) {}
};

struct NonUserDeclaredCopyAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NonUserDeclaredCopyAssignment definition
  // CHECK-NOT: CopyAssignment {{.*}}user_declared{{.*}}
};

struct NeedsImplicitCopyAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NeedsImplicitCopyAssignment definition
  // CHECK: CopyAssignment {{.*}}needs_implicit{{.*}}
  int i = 12;
};

struct DoesNotNeedImplicitCopyAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotNeedImplicitCopyAssignment definition
  // CHECK-NOT: CopyAssignment {{.*}}needs_implicit{{.*}}
  DoesNotNeedImplicitCopyAssignment& operator=(const DoesNotNeedImplicitCopyAssignment&) {}
};

struct CopyAssignmentNeedsOverloadResolution {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct CopyAssignmentNeedsOverloadResolution definition
  // CHECK: CopyAssignment {{.*}}needs_overload_resolution{{.*}}
  mutable int i;
};

struct CopyAssignmentDoesNotNeedOverloadResolution {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct CopyAssignmentDoesNotNeedOverloadResolution definition
  // CHECK-NOT: CopyAssignment {{.*}}needs_overload_resolution{{.*}}
};

struct BaseWithoutCopyAssignmentConstParam {
  BaseWithoutCopyAssignmentConstParam& operator=(BaseWithoutCopyAssignmentConstParam&);
};

struct ImplicitCopyAssignmentHasConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct ImplicitCopyAssignmentHasConstParam definition
  // CHECK: CopyAssignment {{.*}}implicit_has_const_param{{.*}}
};

struct ImplicitCopyAssignmentDoesNotHaveConstParam : BaseWithoutCopyAssignmentConstParam {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct ImplicitCopyAssignmentDoesNotHaveConstParam definition
  // CHECK-NOT: CopyAssignment {{.*}}implicit_has_const_param{{.*}}
};

struct MoveAssignmentExists {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct MoveAssignmentExists definition
  // CHECK: MoveAssignment {{.*}}exists{{.*}}
};

struct MoveAssignmentDoesNotExist {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct MoveAssignmentDoesNotExist definition
  // CHECK-NOT: MoveAssignment {{.*}}exists{{.*}}
  MoveAssignmentDoesNotExist& operator=(const MoveAssignmentDoesNotExist&);
};

struct SimpleMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct SimpleMoveAssignment definition
  // CHECK: MoveAssignment {{.*}}simple{{.*}}
  int i = 12;
};

struct NotSimpleMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NotSimpleMoveAssignment definition
  // CHECK-NOT: MoveAssignment {{.*}}simple{{.*}}
  NotSimpleMoveAssignment& operator=(NotSimpleMoveAssignment&&) = delete;
};

struct TrivialMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct TrivialMoveAssignment definition
  // CHECK: MoveAssignment {{.*}} trivial{{.*}}
  TrivialMoveAssignment() = default;
};

struct NontrivialMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NontrivialMoveAssignment definition
  // CHECK: MoveAssignment {{.*}}non_trivial{{.*}}
  NontrivialMoveAssignment& operator=(NontrivialMoveAssignment&&) {}
};

struct UserDeclaredMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct UserDeclaredMoveAssignment definition
  // CHECK: MoveAssignment {{.*}}user_declared{{.*}}
  UserDeclaredMoveAssignment& operator=(UserDeclaredMoveAssignment&&) {}
};

struct NonUserDeclaredMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NonUserDeclaredMoveAssignment definition
  // CHECK-NOT: MoveAssignment {{.*}}user_declared{{.*}}
};

struct NeedsImplicitMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NeedsImplicitMoveAssignment definition
  // CHECK: MoveAssignment {{.*}}needs_implicit{{.*}}
  int i = 12;
};

struct DoesNotNeedImplicitMoveAssignment {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotNeedImplicitMoveAssignment definition
  // CHECK-NOT: MoveAssignment {{.*}}needs_implicit{{.*}}
  DoesNotNeedImplicitMoveAssignment& operator=(DoesNotNeedImplicitMoveAssignment&&) {}
};

struct MoveAssignmentNeedsOverloadResolution : virtual DeletedDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct MoveAssignmentNeedsOverloadResolution definition
  // CHECK: MoveAssignment {{.*}}needs_overload_resolution{{.*}}
};

struct MoveAssignmentDoesNotNeedOverloadResolution {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct MoveAssignmentDoesNotNeedOverloadResolution definition
  // CHECK-NOT: MoveAssignment {{.*}}needs_overload_resolution{{.*}}
};

struct SimpleDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct SimpleDestructor definition
  // CHECK: Destructor {{.*}}simple{{.*}}
};

struct NotSimpleDestructor : DeletedDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NotSimpleDestructor definition
  // CHECK-NOT: Destructor {{.*}}simple{{.*}}
};

struct IrrelevantDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct IrrelevantDestructor definition
  // CHECK: Destructor {{.*}}irrelevant{{.*}}
};

struct RelevantDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct RelevantDestructor definition
  // CHECK-NOT: Destructor {{.*}}irrelevant{{.*}}
  ~RelevantDestructor() {}
};

struct TrivialDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct TrivialDestructor definition
  // CHECK: Destructor {{.*}} trivial{{.*}}
  ~TrivialDestructor() = default;
};

struct NontrivialDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NontrivialDestructor definition
  // CHECK: Destructor {{.*}}non_trivial{{.*}}
  ~NontrivialDestructor() {}
};

struct UserDeclaredDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct UserDeclaredDestructor definition
  // CHECK: Destructor {{.*}}user_declared{{.*}}
  ~UserDeclaredDestructor() {}
};

struct NonUserDeclaredDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct NonUserDeclaredDestructor definition
  // CHECK-NOT: Destructor {{.*}}user_declared{{.*}}
};

struct NeedsImplicitDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct NeedsImplicitDestructor definition
  // CHECK: Destructor {{.*}}needs_implicit{{.*}}
  int i = 12;
};

struct DoesNotNeedImplicitDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DoesNotNeedImplicitDestructor definition
  // CHECK-NOT: Destructor {{.*}}needs_implicit{{.*}}
  ~DoesNotNeedImplicitDestructor() {}
};

struct DestructorNeedsOverloadResolution : virtual DeletedDestructor {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+3]]:1> line:[[@LINE-1]]:8 struct DestructorNeedsOverloadResolution definition
  // CHECK: Destructor {{.*}}needs_overload_resolution{{.*}}
  ~DestructorNeedsOverloadResolution();
};

struct DestructorDoesNotNeedOverloadResolution {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct DestructorDoesNotNeedOverloadResolution definition
  // CHECK-NOT: Destructor {{.*}}needs_overload_resolution{{.*}}
};

// FIXME: defaulted_is_deleted
