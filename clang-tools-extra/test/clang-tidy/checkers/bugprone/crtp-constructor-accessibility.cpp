// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-crtp-constructor-accessibility %t -- -- -fno-delayed-template-parsing

namespace class_implicit_ctor {
template <typename T>
class CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: CRTP() = default;
// CHECK-FIXES: friend T;

class A : CRTP<A> {};
} // namespace class_implicit_ctor

namespace class_unconstructible {
template <typename T>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class; consider declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: friend T;
    CRTP() = default;
};

class A : CRTP<A> {};
} // namespace class_unconstructible

namespace class_public_default_ctor {
template <typename T>
class CRTP {
public:
    CRTP() = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace class_public_default_ctor

namespace class_public_user_provided_ctor {
template <typename T>
class CRTP {
public:
    CRTP(int) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(int) {}{{[[:space:]]*}}public:
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace class_public_user_provided_ctor

namespace class_public_multiple_user_provided_ctors {
template <typename T>
class CRTP {
public:
    CRTP(int) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(int) {}{{[[:space:]]*}}public:
    CRTP(float) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(float) {}{{[[:space:]]*}}public:
    
    // CHECK-FIXES: friend T;
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace class_public_multiple_user_provided_ctors

namespace class_protected_ctors {
template <typename T>
class CRTP {
protected:
    CRTP(int) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: protected constructor allows the CRTP to be inherited from as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(int) {}{{[[:space:]]*}}protected:
    CRTP() = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: protected constructor allows the CRTP to be inherited from as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}protected:
    CRTP(float) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: protected constructor allows the CRTP to be inherited from as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(float) {}{{[[:space:]]*}}protected:
    
    // CHECK-FIXES: friend T;
    // CHECK-FIXES: friend T;
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace class_protected_ctors

namespace struct_implicit_ctor {
template <typename T>
struct CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:
// CHECK-FIXES: friend T;

class A : CRTP<A> {};
} // namespace struct_implicit_ctor

namespace struct_default_ctor {
template <typename T>
struct CRTP {
    CRTP() = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public constructor allows the CRTP to be constructed as a regular template class; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:
    // CHECK-FIXES: friend T;
};

class A : CRTP<A> {};
} // namespace struct_default_ctor

namespace same_class_multiple_crtps {
template <typename T>
struct CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:
// CHECK-FIXES: friend T;

template <typename T>
struct CRTP2 {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: private:{{[[:space:]]*}}CRTP2() = default;{{[[:space:]]*}}public:
// CHECK-FIXES: friend T;

class A : CRTP<A>, CRTP2<A> {};
} // namespace same_class_multiple_crtps

namespace same_crtp_multiple_classes {
template <typename T>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class; consider declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: friend T;
    CRTP() = default;
};

class A : CRTP<A> {};
class B : CRTP<B> {};
} // namespace same_crtp_multiple_classes

namespace crtp_template {
template <typename T, typename U>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class; consider declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: friend U;
    CRTP() = default;
};

class A : CRTP<int, A> {};
} // namespace crtp_template

namespace crtp_template2 {
template <typename T, typename U>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class; consider declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: friend T;
    CRTP() = default;
};

class A : CRTP<A, A> {};
} // namespace crtp_template2

namespace template_derived {
template <typename T>
class CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: CRTP() = default;
// CHECK-FIXES: friend T;

template<typename T>
class A : CRTP<A<T>> {};

// FIXME: Ideally the warning should be triggered without instantiation.
void foo() {
  A<int> A;
  (void) A;
}
} // namespace template_derived

namespace template_derived_explicit_specialization {
template <typename T>
class CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private and declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: CRTP() = default;
// CHECK-FIXES: friend T;

template<typename T>
class A : CRTP<A<T>> {};

template<>
class A<int> : CRTP<A<int>> {};
} // namespace template_derived_explicit_specialization

namespace explicit_derived_friend {
class A;

template <typename T>
class CRTP {
    CRTP() = default;
    friend A;
};

class A : CRTP<A> {};
} // namespace explicit_derived_friend

namespace explicit_derived_friend_multiple {
class A;

template <typename T>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class; consider declaring the derived class as friend [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: friend T;
    CRTP() = default;
    friend A;
};

class A : CRTP<A> {};
class B : CRTP<B> {};
} // namespace explicit_derived_friend_multiple

namespace no_need_for_friend {
class A;

template <typename T>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible; consider making it private [bugprone-crtp-constructor-accessibility]
// CHECK-FIXES: CRTP() = default;
    friend A;
};

class A : CRTP<A> {};
} // namespace no_need_for_friend

namespace no_warning {
template <typename T>
class CRTP
{
    CRTP() = default;
    friend T;
};

class A : CRTP<A> {};
} // namespace no_warning

namespace no_warning_unsupported {
template<typename... Types>
class CRTP
{};

class A : CRTP<A> {};

void foo() {
    A A;
    (void) A;
}
} // namespace no_warning_unsupported

namespace public_copy_move_constructors_deleted {
template <typename T>
class CRTP
{
    CRTP() = default;
    friend T;
  public:
    CRTP(const CRTP&) = delete;
    CRTP(CRTP&&) = delete;
};

class A : CRTP<A> {};

} // namespace public_copy_move_constructors_deleted

namespace public_copy_protected_move_constructor_deleted {
template <typename T>
class CRTP
{
    CRTP() = default;
    friend T;
  public:
    CRTP(const CRTP&) = delete;
  protected:
    CRTP(CRTP&&) = delete;
};

class A : CRTP<A> {};

} // namespace public_copy_protected_move_constructor_deleted
