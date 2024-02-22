// RUN: %check_clang_tidy %s bugprone-crtp-constructor-accessibility %t -- -- -fno-delayed-template-parsing

namespace class_implicit_ctor {
template <typename T>
class CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider making it private
// CHECK-FIXES: CRTP() = default;

class A : CRTP<A> {};
} // namespace class_implicit_ctor

namespace class_uncostructible {
template <typename T>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider declaring the derived class as friend
// CHECK-FIXES: friend T;
    CRTP() = default;
};

class A : CRTP<A> {};
} // namespace class_uncostructible 

namespace class_public_default_ctor {
template <typename T>
class CRTP {
public:
    CRTP() = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public contructor allows the CRTP to be constructed as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:
};

class A : CRTP<A> {};
} // namespace class_public_default_ctor

namespace class_public_user_provided_ctor {
template <typename T>
class CRTP {
public:
    CRTP(int) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public contructor allows the CRTP to be constructed as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(int) {}{{[[:space:]]*}}public:
};

class A : CRTP<A> {};
} // namespace class_public_user_provided_ctor

namespace class_public_multiple_user_provided_ctors {
template <typename T>
class CRTP {
public:
    CRTP(int) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public contructor allows the CRTP to be constructed as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(int) {}{{[[:space:]]*}}public:
    CRTP(float) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public contructor allows the CRTP to be constructed as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(float) {}{{[[:space:]]*}}public:
};

class A : CRTP<A> {};
} // namespace class_public_multiple_user_provided_ctors

namespace class_protected_ctors {
template <typename T>
class CRTP {
protected:
    CRTP(int) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: protected contructor allows the CRTP to be inherited from as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(int) {}{{[[:space:]]*}}protected:
    CRTP() = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: protected contructor allows the CRTP to be inherited from as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}protected:
    CRTP(float) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: protected contructor allows the CRTP to be inherited from as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP(float) {}{{[[:space:]]*}}protected:
};

class A : CRTP<A> {};
} // namespace class_protected_ctors

namespace struct_implicit_ctor {
template <typename T>
struct CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: the implicit default constructor of the CRTP is publicly accessible [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:8: note: consider making it private
// CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:

class A : CRTP<A> {};
} // namespace struct_implicit_ctor

namespace struct_default_ctor {
template <typename T>
struct CRTP {
    CRTP() = default;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: public contructor allows the CRTP to be constructed as a regular template class [bugprone-crtp-constructor-accessibility]
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: consider making it private
    // CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:
};

class A : CRTP<A> {};
} // namespace struct_default_ctor

namespace same_class_multiple_crtps {
template <typename T>
struct CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: the implicit default constructor of the CRTP is publicly accessible [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:8: note: consider making it private
// CHECK-FIXES: private:{{[[:space:]]*}}CRTP() = default;{{[[:space:]]*}}public:

template <typename T>
struct CRTP2 {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: the implicit default constructor of the CRTP is publicly accessible [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:8: note: consider making it private
// CHECK-FIXES: private:{{[[:space:]]*}}CRTP2() = default;{{[[:space:]]*}}public:

class A : CRTP<A>, CRTP2<A> {};
} // namespace same_class_multiple_crtps

namespace same_crtp_multiple_classes {
template <typename T>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider declaring the derived class as friend
// CHECK-FIXES: friend T;
    CRTP() = default;
};

class A : CRTP<A> {};
class B : CRTP<B> {};
} // namespace same_crtp_multiple_classes

namespace crtp_template {
template <typename T, typename U>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider declaring the derived class as friend
// CHECK-FIXES: friend U;
    CRTP() = default;
};

class A : CRTP<int, A> {};
} // namespace crtp_template

namespace crtp_template2 {
template <typename T, typename U>
class CRTP {
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider declaring the derived class as friend
// CHECK-FIXES: friend T;
    CRTP() = default;
};

class A : CRTP<A, A> {};
} // namespace crtp_template2

namespace template_derived {
template <typename T>
class CRTP {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider making it private
// CHECK-FIXES: CRTP() = default;

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
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the implicit default constructor of the CRTP is publicly accessible [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider making it private
// CHECK-FIXES: CRTP() = default;

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
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: the CRTP cannot be constructed from the derived class [bugprone-crtp-constructor-accessibility]
// CHECK-MESSAGES: :[[@LINE-2]]:7: note: consider declaring the derived class as friend
// CHECK-FIXES: friend T;
    CRTP() = default;
    friend A;
};

class A : CRTP<A> {};
class B : CRTP<B> {};
} // namespace explicit_derived_friend_multiple

namespace no_warning {
template <typename T>
class CRTP
{
    CRTP() = default;
    friend T;
};

class A : CRTP<A> {};
} // namespace no_warning
