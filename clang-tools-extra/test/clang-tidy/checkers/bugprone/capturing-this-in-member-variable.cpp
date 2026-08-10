// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-capturing-this-in-member-variable %t -- -config="{CheckOptions: {bugprone-capturing-this-in-member-variable.FunctionWrapperTypes: '::std::function;::Fn', bugprone-capturing-this-in-member-variable.BindFunctions: '::std::bind;::Bind'}}" --

namespace std {

template<class Fn>
class function;

template<class R, class ...Args>
class function<R(Args...)> {
public:
  function() noexcept;
  template<class F> function(F &&);
};

template <typename F, typename... Args>
function<F(Args...)> bind(F&&, Args&&...) {
  return {};
}

} // namespace std

struct Fn {
  template<class F> Fn(F &&);
};

template <typename F, typename... Args>
std::function<F(Args...)> Bind(F&&, Args&&...) {
  return {};
}

struct BasicConstructor {
  BasicConstructor() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: 'this' captured by a lambda and stored in a class member variable;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BasicConstructorWithCaptureAllByValue {
  BasicConstructorWithCaptureAllByValue() : Captured([=]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: 'this' captured by a lambda and stored in a class member variable;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BasicConstructorWithCaptureAllByRef {
  BasicConstructorWithCaptureAllByRef() : Captured([&]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: 'this' captured by a lambda and stored in a class member variable;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BasicField1 {
  std::function<void()> Captured = [this]() { static_cast<void>(this); };
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: 'this' captured by a lambda and stored in a class member variable;
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};
struct BasicField2 {
  std::function<void()> Captured{[this]() { static_cast<void>(this); }};
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: 'this' captured by a lambda and stored in a class member variable;
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct NotCaptureThis {
  NotCaptureThis(int V) : Captured([V]() { static_cast<void>(V); }) {}
  std::function<void()> Captured;
};

struct AssignCapture {
  AssignCapture() : Captured([Self = this]() { static_cast<void>(Self); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: 'this' captured by a lambda and stored in a class member variable;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
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
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: 'this' captured by a lambda and stored in a class member variable;
  DeleteCopyDefaultMove(DeleteCopyDefaultMove const&) = delete;
  DeleteCopyDefaultMove(DeleteCopyDefaultMove &&) = default;
  DeleteCopyDefaultMove& operator=(DeleteCopyDefaultMove const&) = delete;
  DeleteCopyDefaultMove& operator=(DeleteCopyDefaultMove &&) = default;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct DeleteMoveDefaultCopy {
  DeleteMoveDefaultCopy() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: 'this' captured by a lambda and stored in a class member variable;
  DeleteMoveDefaultCopy(DeleteMoveDefaultCopy const&) = default;
  DeleteMoveDefaultCopy(DeleteMoveDefaultCopy &&) = delete;
  DeleteMoveDefaultCopy& operator=(DeleteMoveDefaultCopy const&) = default;
  DeleteMoveDefaultCopy& operator=(DeleteMoveDefaultCopy &&) = delete;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct DeleteCopyBase {
  DeleteCopyBase() = default;
  DeleteCopyBase(DeleteCopyBase const&) = delete;
  DeleteCopyBase(DeleteCopyBase &&) = default;
  DeleteCopyBase& operator=(DeleteCopyBase const&) = delete;
  DeleteCopyBase& operator=(DeleteCopyBase &&) = default;
};

struct DeleteMoveBase {
  DeleteMoveBase() = default;
  DeleteMoveBase(DeleteMoveBase const&) = default;
  DeleteMoveBase(DeleteMoveBase &&) = delete;
  DeleteMoveBase& operator=(DeleteMoveBase const&) = default;
  DeleteMoveBase& operator=(DeleteMoveBase &&) = delete;
};

struct DeleteCopyMoveBase : DeleteCopyBase, DeleteMoveBase {};

struct InheritDeleteCopy : DeleteCopyBase {
  InheritDeleteCopy() : DeleteCopyBase{}, Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: 'this' captured by a lambda and stored in a class member variable;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};
struct InheritDeleteMove : DeleteMoveBase {
  InheritDeleteMove() : DeleteMoveBase{}, Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: 'this' captured by a lambda and stored in a class member variable;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};
struct InheritDeleteCopyMove : DeleteCopyMoveBase {
  InheritDeleteCopyMove() : DeleteCopyMoveBase{}, Captured([this]() { static_cast<void>(this); }) {}
  std::function<void()> Captured;
};

struct PrivateCopyMoveBase {
// It is how to disable copy and move in C++03
  PrivateCopyMoveBase() = default;
private:
  PrivateCopyMoveBase(PrivateCopyMoveBase const&) = default;
  PrivateCopyMoveBase(PrivateCopyMoveBase &&) = default;
  PrivateCopyMoveBase& operator=(PrivateCopyMoveBase const&) = default;
  PrivateCopyMoveBase& operator=(PrivateCopyMoveBase &&) = default;
};
struct InheritPrivateCopyMove : PrivateCopyMoveBase {
  InheritPrivateCopyMove() : PrivateCopyMoveBase{}, Captured([this]() { static_cast<void>(this); }) {}
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
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: 'this' captured by a lambda and stored in a class member variable;
  UserDefinedCopyMoveWithDefault1(UserDefinedCopyMoveWithDefault1 const&) = default;
  UserDefinedCopyMoveWithDefault1(UserDefinedCopyMoveWithDefault1 &&);
  UserDefinedCopyMoveWithDefault1& operator=(UserDefinedCopyMoveWithDefault1 const&);
  UserDefinedCopyMoveWithDefault1& operator=(UserDefinedCopyMoveWithDefault1 &&);
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct UserDefinedCopyMoveWithDefault2 {
  UserDefinedCopyMoveWithDefault2() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: 'this' captured by a lambda and stored in a class member variable;
  UserDefinedCopyMoveWithDefault2(UserDefinedCopyMoveWithDefault2 const&);
  UserDefinedCopyMoveWithDefault2(UserDefinedCopyMoveWithDefault2 &&) = default;
  UserDefinedCopyMoveWithDefault2& operator=(UserDefinedCopyMoveWithDefault2 const&);
  UserDefinedCopyMoveWithDefault2& operator=(UserDefinedCopyMoveWithDefault2 &&);
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct UserDefinedCopyMoveWithDefault3 {
  UserDefinedCopyMoveWithDefault3() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: 'this' captured by a lambda and stored in a class member variable;
  UserDefinedCopyMoveWithDefault3(UserDefinedCopyMoveWithDefault3 const&);
  UserDefinedCopyMoveWithDefault3(UserDefinedCopyMoveWithDefault3 &&);
  UserDefinedCopyMoveWithDefault3& operator=(UserDefinedCopyMoveWithDefault3 const&) = default;
  UserDefinedCopyMoveWithDefault3& operator=(UserDefinedCopyMoveWithDefault3 &&);
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct UserDefinedCopyMoveWithDefault4 {
  UserDefinedCopyMoveWithDefault4() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: 'this' captured by a lambda and stored in a class member variable;
  UserDefinedCopyMoveWithDefault4(UserDefinedCopyMoveWithDefault4 const&);
  UserDefinedCopyMoveWithDefault4(UserDefinedCopyMoveWithDefault4 &&);
  UserDefinedCopyMoveWithDefault4& operator=(UserDefinedCopyMoveWithDefault4 const&);
  UserDefinedCopyMoveWithDefault4& operator=(UserDefinedCopyMoveWithDefault4 &&) = default;
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct CustomFunctionWrapper {
  CustomFunctionWrapper() : Captured([this]() { static_cast<void>(this); }) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: 'this' captured by a lambda and stored in a class member variable;
  Fn Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: note: class member of type 'Fn' that stores captured 'this'
};

struct BindConstructor {
  BindConstructor() : Captured(std::bind(&BindConstructor::method, this)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: 'this' captured by a 'std::bind' call and stored in a class member variable;
  void method() {}
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BindField1 {
  void method() {}
  std::function<void()> Captured = std::bind(&BindField1::method, this);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: 'this' captured by a 'std::bind' call and stored in a class member variable;
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BindField2 {
  void method() {}
  std::function<void()> Captured{std::bind(&BindField2::method, this)};
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: 'this' captured by a 'std::bind' call and stored in a class member variable;
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BindCustom {
  BindCustom() : Captured(Bind(&BindCustom::method, this)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: 'this' captured by a 'Bind' call and stored in a class member variable;
  void method() {}
  std::function<void()> Captured;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: note: class member of type 'std::function<void (void)>' that stores captured 'this'
};

struct BindNotCapturingThis {
  void method(int) {}
  BindNotCapturingThis(int V) : Captured(std::bind(&BindNotCapturingThis::method, V)) {}
  std::function<void()> Captured;
};

struct DeletedCopyMoveWithBind {
  DeletedCopyMoveWithBind() : Captured(std::bind(&DeletedCopyMoveWithBind::method, this)) {}
  DeletedCopyMoveWithBind(DeletedCopyMoveWithBind const&) = delete;
  DeletedCopyMoveWithBind(DeletedCopyMoveWithBind &&) = delete;
  DeletedCopyMoveWithBind& operator=(DeletedCopyMoveWithBind const&) = delete;
  DeletedCopyMoveWithBind& operator=(DeletedCopyMoveWithBind &&) = delete;
  void method() {}
  std::function<void()> Captured;
};
