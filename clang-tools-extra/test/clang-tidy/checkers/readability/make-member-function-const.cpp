// RUN: %check_clang_tidy %s readability-make-member-function-const %t

struct Str {
  void const_method() const;
  void non_const_method();
};

namespace Diagnose {
struct A;

void free_const_use(const A *);
void free_const_use(const A &);

struct A {
  int M;
  const int ConstM;
  struct {
    int M;
  } Struct;
  Str S;
  Str &Sref;

  void already_const() const;

  int read_field() {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'read_field' can be made const
    // CHECK-FIXES: int read_field() const {
    return M;
  }

  int read_struct_field() {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'read_struct_field' can be made const
    // CHECK-FIXES: int read_struct_field() const {
    return Struct.M;
  }

  int read_const_field() {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'read_const_field' can be made const
    // CHECK-FIXES: int read_const_field() const {
    return ConstM;
  }

  int read_fields_in_parentheses() {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'read_fields_in_parentheses' can be made const
    // CHECK-FIXES: int read_fields_in_parentheses() const {
    return (this)->M + (((((Struct.M))))) + ((this->ConstM));
  }

  void call_const_member() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'call_const_member' can be made const
    // CHECK-FIXES: void call_const_member() const {
    already_const();
  }

  void call_const_member_on_public_field() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'call_const_member_on_public_field' can be made const
    // CHECK-FIXES: void call_const_member_on_public_field() const {
    S.const_method();
  }

  void call_const_member_on_public_field_ref() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'call_const_member_on_public_field_ref' can be made const
    // CHECK-FIXES: void call_const_member_on_public_field_ref() const {
    Sref.const_method();
  }

  const Str &return_public_field_ref() {
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: method 'return_public_field_ref' can be made const
    // CHECK-FIXES: const Str &return_public_field_ref() const {
    return S;
  }

  const A *return_this_const() {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: method 'return_this_const' can be made const
    // CHECK-FIXES: const A *return_this_const() const {
    return this;
  }

  const A &return_this_const_ref() {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: method 'return_this_const_ref' can be made const
    // CHECK-FIXES: const A &return_this_const_ref() const {
    return *this;
  }

  void const_use() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'const_use' can be made const
    // CHECK-FIXES: void const_use() const {
    free_const_use(this);
  }

  void const_use_ref() {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'const_use_ref' can be made const
    // CHECK-FIXES: void const_use_ref() const {
    free_const_use(*this);
  }

  auto trailingReturn() -> int {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: method 'trailingReturn' can be made const
    // CHECK-FIXES: auto trailingReturn() const -> int {
    return M;
  }

  int volatileFunction() volatile {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'volatileFunction' can be made const
    // CHECK-FIXES: int volatileFunction() const volatile {
    return M;
  }

  int restrictFunction() __restrict {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'restrictFunction' can be made const
    // CHECK-FIXES: int restrictFunction() const __restrict {
    return M;
  }

  int refFunction() & {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: method 'refFunction' can be made const
    // CHECK-FIXES: int refFunction() const & {
    return M;
  }

  void out_of_line_call_const();
  // CHECK-FIXES: void out_of_line_call_const() const;
};

void A::out_of_line_call_const() {
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: method 'out_of_line_call_const' can be made const
  // CHECK-FIXES: void A::out_of_line_call_const() const {
  already_const();
}
} // namespace Diagnose

namespace Keep {
struct Keep;
void free_non_const_use(Keep *);
void free_non_const_use(Keep &);

struct Keep {
private:
  void private_const_method() const;
  Str PrivateS;
  Str *Sptr;
  Str &Sref;

public:
  int M;
  Str S;

  void keepTrivial() {}

  // See readability-convert-member-functions-to-static instead.
  void keepStatic() { int I = 0; }

  const int *keepConstCast() const;
  int *keepConstCast() { // Needs to stay non-const.
    return const_cast<int *>(static_cast<const Keep *>(this)->keepConstCast());
  }

  void non_const_use() { free_non_const_use(this); }
  void non_const_use_ref() { free_non_const_use(*this); }

  Keep *return_this() {
    return this;
  }

  Keep &return_this_ref() {
    return *this;
  }

  void escape_this() {
    Keep *Escaped = this;
  }

  void call_private_const_method() {
    private_const_method();
  }

  int keepConst() const { return M; }

  virtual int keepVirtual() { return M; }

  void writeField() {
    M = 1;
  }

  void callNonConstMember() { writeField(); }

  void call_non_const_member_on_field() { S.non_const_method(); }

  void call_const_member_on_private_field() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    PrivateS.const_method();
  }

  const Str &return_private_field_ref() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    return PrivateS;
  }

  void call_non_const_member_on_pointee() {
    Sptr->non_const_method();
  }

  void call_const_member_on_pointee() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    Sptr->const_method();
  }

  Str *return_pointer() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    return Sptr;
  }

  const Str *return_const_pointer() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    return Sptr;
  }

  void call_non_const_member_on_ref() {
    Sref.non_const_method();
  }

  void escaped_private_field() {
    const auto &Escaped = Sref;
  }

  Str &return_field_ref() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    return Sref;
  }

  const Str &return_field_const_ref() {
    // Technically, this method could be const-qualified,
    // but it might not be logically const.
    return Sref;
  }
};

struct KeepVirtualDerived : public Keep {
  int keepVirtual() { return M; }
};

void KeepLambdas() {
  auto F = +[]() { return 0; };
  auto F2 = []() { return 0; };
}

template <class Base>
struct KeepWithDependentBase : public Base {
  int M;
  // We cannot make this method const because it might need to override
  // a function from Base.
  int const_f() { return M; }
};

template <class T>
struct KeepClassTemplate {
  int M;
  // We cannot make this method const because a specialization
  // might use *this differently.
  int const_f() { return M; }
};

struct KeepMemberFunctionTemplate {
  int M;
  // We cannot make this method const because a specialization
  // might use *this differently.
  template <class T>
  int const_f() { return M; }
};

void instantiate() {
  struct S {};
  KeepWithDependentBase<S> I1;
  I1.const_f();

  KeepClassTemplate<int> I2;
  I2.const_f();

  KeepMemberFunctionTemplate I3;
  I3.const_f<int>();
}

struct NoFixitInMacro {
  int M;

#define FUN const_use_macro()
  int FUN {
    return M;
  }

#define T(FunctionName, Keyword) \
  int FunctionName() Keyword { return M; }
#define EMPTY
  T(A, EMPTY)
  T(B, const)

#define T2(FunctionName) \
  int FunctionName() { return M; }
  T2(A2)
};

// Real-world code, see clang::ObjCInterfaceDecl.
class DataPattern {
  int &data() const;

public:
  const int &get() const {
    return const_cast<DataPattern *>(this)->get();
  }

  // This member function must stay non-const, even though
  // it only calls other private const member functions.
  int &get() {
    return data();
  }

  void set() {
    data() = 42;
  }
};

struct MemberFunctionPointer {
  void call_non_const(void (MemberFunctionPointer::*FP)()) {
    (this->*FP)();
  }

  void call_const(void (MemberFunctionPointer::*FP)() const) {
    (this->*FP)();
  }
};

} // namespace Keep
