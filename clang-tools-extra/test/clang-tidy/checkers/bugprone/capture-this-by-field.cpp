// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-capture-this-by-field %t

namespace std {

template<class Fn>
class function;

template<class R, class ...Args>
class function<R(Args...)> {
public:
  function() noexcept;
  template<class F> function(F &&);
};

} // namespace std

struct Basic {
  Basic() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: using lambda expressions to capture this and storing it in class member
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct AssignCapture {
  AssignCapture() : Captured([Self = this]() { static_cast<void>(Self); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: using lambda expressions to capture this and storing it in class member
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct DeleteMoveAndCopy {
  DeleteMoveAndCopy() : Captured([this]() { static_cast<void>(this); }) {}
  DeleteMoveAndCopy(DeleteMoveAndCopy const&) = delete;
  DeleteMoveAndCopy(DeleteMoveAndCopy &&) = delete;
  DeleteMoveAndCopy& operator=(DeleteMoveAndCopy const&) = delete;
  DeleteMoveAndCopy& operator=(DeleteMoveAndCopy &&) = delete;
  std::function<void()> Captured;
};

struct DeleteCopyImplicitDisabledMove {
  DeleteCopyImplicitDisabledMove() : Captured([this]() { static_cast<void>(this); }) {}
  DeleteCopyImplicitDisabledMove(DeleteCopyImplicitDisabledMove const&) = delete;
  DeleteCopyImplicitDisabledMove& operator=(DeleteCopyImplicitDisabledMove const&) = delete;
  std::function<void()> Captured;
};

struct DeleteCopyDefaultMove {
  DeleteCopyDefaultMove() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: using lambda expressions to capture this and storing it in class member
  DeleteCopyDefaultMove(DeleteCopyDefaultMove const&) = delete;
  DeleteCopyDefaultMove(DeleteCopyDefaultMove &&) = default;
  DeleteCopyDefaultMove& operator=(DeleteCopyDefaultMove const&) = delete;
  DeleteCopyDefaultMove& operator=(DeleteCopyDefaultMove &&) = default;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct DeleteMoveDefaultCopy {
  DeleteMoveDefaultCopy() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: using lambda expressions to capture this and storing it in class member
  DeleteMoveDefaultCopy(DeleteMoveDefaultCopy const&) = default;
  DeleteMoveDefaultCopy(DeleteMoveDefaultCopy &&) = delete;
  DeleteMoveDefaultCopy& operator=(DeleteMoveDefaultCopy const&) = default;
  DeleteMoveDefaultCopy& operator=(DeleteMoveDefaultCopy &&) = delete;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct DeleteCopyMoveBase {
  DeleteCopyMoveBase() = default;
  DeleteCopyMoveBase(DeleteCopyMoveBase const&) = delete;
  DeleteCopyMoveBase(DeleteCopyMoveBase &&) = delete;
  DeleteCopyMoveBase& operator=(DeleteCopyMoveBase const&) = delete;
  DeleteCopyMoveBase& operator=(DeleteCopyMoveBase &&) = delete;
};

struct Inherit : DeleteCopyMoveBase {
  Inherit() : DeleteCopyMoveBase{}, Captured([this]() { static_cast<void>(this); }) {}
  std::function<void()> Captured;
};

struct UserDefinedCopyMove {
  UserDefinedCopyMove() : Captured([this]() { static_cast<void>(this); }) {}
  UserDefinedCopyMove(UserDefinedCopyMove const&);
  UserDefinedCopyMove(UserDefinedCopyMove &&);
  UserDefinedCopyMove& operator=(UserDefinedCopyMove const&);
  UserDefinedCopyMove& operator=(UserDefinedCopyMove &&);
  std::function<void()> Captured;
};

struct UserDefinedCopyMoveWithDefault1 {
  UserDefinedCopyMoveWithDefault1() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: using lambda expressions to capture this and storing it in class member
  UserDefinedCopyMoveWithDefault1(UserDefinedCopyMoveWithDefault1 const&) = default;
  UserDefinedCopyMoveWithDefault1(UserDefinedCopyMoveWithDefault1 &&);
  UserDefinedCopyMoveWithDefault1& operator=(UserDefinedCopyMoveWithDefault1 const&);
  UserDefinedCopyMoveWithDefault1& operator=(UserDefinedCopyMoveWithDefault1 &&);
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct UserDefinedCopyMoveWithDefault2 {
  UserDefinedCopyMoveWithDefault2() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: using lambda expressions to capture this and storing it in class member
  UserDefinedCopyMoveWithDefault2(UserDefinedCopyMoveWithDefault2 const&);
  UserDefinedCopyMoveWithDefault2(UserDefinedCopyMoveWithDefault2 &&) = default;
  UserDefinedCopyMoveWithDefault2& operator=(UserDefinedCopyMoveWithDefault2 const&);
  UserDefinedCopyMoveWithDefault2& operator=(UserDefinedCopyMoveWithDefault2 &&);
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct UserDefinedCopyMoveWithDefault3 {
  UserDefinedCopyMoveWithDefault3() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: using lambda expressions to capture this and storing it in class member
  UserDefinedCopyMoveWithDefault3(UserDefinedCopyMoveWithDefault3 const&);
  UserDefinedCopyMoveWithDefault3(UserDefinedCopyMoveWithDefault3 &&);
  UserDefinedCopyMoveWithDefault3& operator=(UserDefinedCopyMoveWithDefault3 const&) = default;
  UserDefinedCopyMoveWithDefault3& operator=(UserDefinedCopyMoveWithDefault3 &&);
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};

struct UserDefinedCopyMoveWithDefault4 {
  UserDefinedCopyMoveWithDefault4() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: using lambda expressions to capture this and storing it in class member
  UserDefinedCopyMoveWithDefault4(UserDefinedCopyMoveWithDefault4 const&);
  UserDefinedCopyMoveWithDefault4(UserDefinedCopyMoveWithDefault4 &&);
  UserDefinedCopyMoveWithDefault4& operator=(UserDefinedCopyMoveWithDefault4 const&);
  UserDefinedCopyMoveWithDefault4& operator=(UserDefinedCopyMoveWithDefault4 &&) = default;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: 'std::function' that stores captured this
};
