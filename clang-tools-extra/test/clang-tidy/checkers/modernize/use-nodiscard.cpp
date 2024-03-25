// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-use-nodiscard %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-nodiscard.ReplacementString: 'NO_DISCARD'}}"

namespace std {
template <class>
class function;
class string {};
}

namespace boost {
template <class>
class function;
}

#define MUST_USE_RESULT __attribute__((warn_unused_result))
#define NO_DISCARD [[nodiscard]]
#define NO_RETURN [[noreturn]]

#define BOOLEAN_FUNC bool f23() const

typedef unsigned my_unsigned;
typedef unsigned &my_unsigned_reference;
typedef const unsigned &my_unsigned_const_reference;

struct NO_DISCARD NoDiscardStruct{};

class Foo {
public:
    using size_type = unsigned;

    bool f1() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f1' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f1() const;

    bool f2(int) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f2' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f2(int) const;

    bool f3(const int &) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f3' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f3(const int &) const;

    bool f4(void) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f4' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f4(void) const;
    
    // negative tests

    void f5() const;
    
    bool f6();
    
    bool f7(int &);
    
    bool f8(int &) const;
    
    bool f9(int *) const;
    
    bool f10(const int &, int &) const;
    
    NO_DISCARD bool f12() const;
    
    MUST_USE_RESULT bool f13() const;
    
    [[nodiscard]] bool f11() const;
    
    [[clang::warn_unused_result]] bool f11a() const;
    
    [[gnu::warn_unused_result]] bool f11b() const;

    bool _f20() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function '_f20' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool _f20() const;
    
    NO_RETURN bool f21() const;
    
    ~Foo();
    
    bool operator+=(int) const;
    
    // extra keywords (virtual,inline,const) on return type
    
    virtual bool f14() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f14' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD virtual bool f14() const;
    
    const bool f15() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f15' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD const bool f15() const;
    
    inline const bool f16() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f16' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD inline const bool f16() const;

    inline const std::string &f45() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f45' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD inline const std::string &f45() const;

    inline virtual const bool f17() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f17' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD inline virtual const bool f17() const;

    // inline with body
    bool f18() const 
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f18' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f18() const 
    {
     return true;
    }

    bool f19() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f19' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f19() const;

    BOOLEAN_FUNC;
    
    bool f24(size_type) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f24' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f24(size_type) const;
    
    bool f28(my_unsigned) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f28' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f28(my_unsigned) const;

    bool f29(my_unsigned_reference) const;

    bool f30(my_unsigned_const_reference) const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f30' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool f30(my_unsigned_const_reference) const;

    template <class F>
    F f37(F a, F b) const;

    template <class F>
    bool f38(F a) const;

    bool f39(const std::function<bool()> &predicate) const;

    bool f39a(std::function<bool()> predicate) const;

    bool f39b(const std::function<bool()> predicate) const;

    bool f45(const boost::function<bool()> &predicate) const;

    bool f45a(boost::function<bool()> predicate) const;

    bool f45b(const boost::function<bool()> predicate) const;

    // Do not add ``[[nodiscard]]`` to parameter packs.
    template <class... Args>
    bool ParameterPack(Args... args) const;

    template <typename... Targs>
    bool ParameterPack2(Targs... Fargs) const;

    // Do not add ``[[nodiscard]]`` to variadic functions.
    bool VariadicFunctionTest(const int &, ...) const;

    // Do not add ``[[nodiscard]]`` to non constant static functions.
    static bool not_empty();

    // Do not add ``[[nodiscard]]`` to conversion functions.
    // explicit operator bool() const { return true; }

    // Do not add ``[[nodiscard]]`` to functions returning types marked [[nodiscard]].
    NoDiscardStruct f50() const;
};

// Do not add ``[[nodiscard]]`` to Lambda.
const auto nonConstReferenceType = [] {
  return true;
};

auto lambda1 = [](int a, int b) { return a < b; };
auto lambda1a = [](int a) { return a; };
auto lambda1b = []()  { return true;};

auto get_functor = [](bool check) {
    return  [&](const std::string& sr)->std::string {
        if(check){
            return std::string();
        }
        return std::string();
    };
};

// Do not add ``[[nodiscard]]`` to function definition.
bool Foo::f19() const {
  return true;
}

template <class T>
class Bar {
public:
    using value_type = T;
    using reference = value_type &;
    using const_reference = const value_type &;

    // Do not add ``[[nodiscard]]`` to non explicit conversion functions.
    operator bool() const { return true; }

    bool empty() const;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'empty' should be marked NO_DISCARD [modernize-use-nodiscard]
    // CHECK-FIXES: NO_DISCARD bool empty() const;

    // we cannot assume that the template parameter isn't a pointer
    bool f25(value_type) const;

    bool f27(reference) const;

    typename T::value_type f35() const;

    T f34() const;

    bool f31(T) const;

    bool f33(T &) const;

    bool f26(const_reference) const;

    bool f32(const T &) const;
};

template <typename _Tp, int cn>
class Vec {
public:
    Vec(_Tp v0, _Tp v1); //!< 2-element vector constructor

    Vec cross(const Vec &v) const;

    template <typename T2>
    operator Vec<T2, cn>() const;
};
    
template <class T>
class Bar2 {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;

  // we cannot assume that the template parameter isn't a pointer
  bool f40(value_type) const;

  bool f41(reference) const;

  value_type f42() const;

  typename T::value_type f43() const;

  bool f44(const_reference) const;
};

template <class T>
bool Bar<T>::empty() const {
  return true;
}

// don't mark typical ``[[nodiscard]]`` candidates if the class
// has mutable member variables
class MutableExample {
  mutable bool m_isempty;

public:
  bool empty() const;
};
