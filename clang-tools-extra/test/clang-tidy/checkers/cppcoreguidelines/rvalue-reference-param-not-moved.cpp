// RUN: %check_clang_tidy -std=c++11 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowPartialMove: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreUnnamedParams: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreNonDeducedTemplateTypes: true}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffix=,CXX14 -std=c++14 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowPartialMove: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreUnnamedParams: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreNonDeducedTemplateTypes: true}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffix=,NOSUBEXPR -std=c++11 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowPartialMove: false, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreUnnamedParams: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreNonDeducedTemplateTypes: true}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffix=,UNNAMED -std=c++11 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowPartialMove: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreUnnamedParams: false, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreNonDeducedTemplateTypes: true}}" -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffix=,NONDEDUCED -std=c++11 %s cppcoreguidelines-rvalue-reference-param-not-moved %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-rvalue-reference-param-not-moved.AllowPartialMove: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreUnnamedParams: true, cppcoreguidelines-rvalue-reference-param-not-moved.IgnoreNonDeducedTemplateTypes: false}}" -- -fno-delayed-template-parsing

// NOLINTBEGIN
namespace std {
template <typename>
struct remove_reference;

template <typename _Tp> struct remove_reference { typedef _Tp type; };
template <typename _Tp> struct remove_reference<_Tp&> { typedef _Tp type; };
template <typename _Tp> struct remove_reference<_Tp&&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) noexcept;

template <typename _Tp>
constexpr _Tp &&
forward(typename remove_reference<_Tp>::type &__t) noexcept;

}
// NOLINTEND

struct Obj {
  Obj();
  Obj(const Obj&);
  Obj& operator=(const Obj&);
  Obj(Obj&&);
  Obj& operator=(Obj&&);
  void member() const;
};

void consumes_object(Obj);

void never_moves_param(Obj&& o) {
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  o.member();
}

void just_a_declaration(Obj&& o);

void copies_object(Obj&& o) {
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  Obj copy = o;
}

template <typename T>
void never_moves_param_template(Obj&& o, T t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  o.member();
}

void never_moves_params(Obj&& o1, Obj&& o2) {
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: rvalue reference parameter 'o1' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  // CHECK-MESSAGES: :[[@LINE-2]]:41: warning: rvalue reference parameter 'o2' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
}

void never_moves_some_params(Obj&& o1, Obj&& o2) {
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: rvalue reference parameter 'o1' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]

  Obj other{std::move(o2)};
}

void never_moves_unnamed(Obj&&) {}
// CHECK-MESSAGES-UNNAMED: :[[@LINE-1]]:31: warning: rvalue reference parameter '' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]

void never_moves_mixed(Obj o1, Obj&& o2) {
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: rvalue reference parameter 'o2' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
}

void lambda_captures_parameter_as_value(Obj&& o) {
  auto f = [o]() {
    consumes_object(std::move(o));
  };
  // CHECK-MESSAGES: :[[@LINE-4]]:47: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
}

void lambda_captures_parameter_as_value_nested(Obj&& o) {
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  auto f = [&o]() {
    auto f_nested = [o]() {
      consumes_object(std::move(o));
    };
  };
  auto f2 = [o]() {
    auto f_nested = [&o]() {
      consumes_object(std::move(o));
    };
  };
  auto f3 = [o]() {
    auto f_nested = [&o]() {
      auto f_nested_inner = [&o]() {
        consumes_object(std::move(o));
      };
    };
  };
  auto f4 = [&o]() {
    auto f_nested = [&o]() {
      auto f_nested_inner = [o]() {
        consumes_object(std::move(o));
      };
    };
  };
}

void misc_lambda_checks() {
  auto never_moves = [](Obj&& o1) {
    Obj other{o1};
  };
  // CHECK-MESSAGES: :[[@LINE-3]]:31: warning: rvalue reference parameter 'o1' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]

#if __cplusplus >= 201402L
  auto never_moves_with_auto_param = [](Obj&& o1, auto& v) {
    Obj other{o1};
  };
  // CHECK-MESSAGES-CXX14: :[[@LINE-3]]:47: warning: rvalue reference parameter 'o1' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
#endif
}

template <typename T>
void forwarding_ref(T&& t) {
  t.member();
}

template <typename T>
void forwarding_ref_forwarded(T&& t) {
  forwarding_ref(std::forward<T>(t));
}

template <typename... Ts>
void type_pack(Ts&&... ts) {
  (forwarding_ref(std::forward<Ts>(ts)), ...);
}

void call_forwarding_functions() {
  Obj o;

  forwarding_ref(Obj{});
  type_pack(Obj{});
  type_pack(Obj{}, o);
  type_pack(Obj{}, Obj{});
}

void moves_parameter(Obj&& o) {
  Obj moved = std::move(o);
}

void moves_parameter_extra_parens(Obj&& o) {
  Obj moved = std::move((o));
}

void does_not_move_in_evaluated(Obj&& o) {
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  using result_t = decltype(std::move(o));
  unsigned size = sizeof(std::move(o));
  Obj moved = o;
}

template <typename T1, typename T2>
struct mypair {
  T1 first;
  T2 second;
};

void moves_member_of_parameter(mypair<Obj, Obj>&& pair) {
  // CHECK-MESSAGES-NOSUBEXPR: :[[@LINE-1]]:51: warning: rvalue reference parameter 'pair' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  Obj a = std::move(pair.first);
  Obj b = std::move(pair.second);
}

template <typename T>
struct myoptional {
  T& operator*() &;
  T&& operator*() &&;
};

void moves_deref_optional(myoptional<Obj>&& opt) {
  // CHECK-MESSAGES-NOSUBEXPR: :[[@LINE-1]]:45: warning: rvalue reference parameter 'opt' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
  Obj other = std::move(*opt);
}

void moves_optional_then_deref_resulting_rvalue(myoptional<Obj>&& opt) {
  Obj other = *std::move(opt);
}

void pass_by_lvalue_reference(Obj& o) {
  o.member();
}

void pass_by_value(Obj o) {
  o.member();
}

void pass_by_const_lvalue_reference(const Obj& o) {
  o.member();
}

void pass_by_const_lvalue_reference(const Obj&& o) {
  o.member();
}

void lambda_captures_parameter_as_reference(Obj&& o) {
  auto f = [&o]() {
    consumes_object(std::move(o));
  };
}

void lambda_captures_parameter_as_reference_nested(Obj&& o) {
  auto f = [&o]() {
    auto f_nested = [&o]() {
      auto f_nested2 = [&o]() {
        consumes_object(std::move(o));
      };
    };
  };
}

#if __cplusplus >= 201402L
void lambda_captures_parameter_generalized(Obj&& o) {
  auto f = [o = std::move(o)]() {
    consumes_object(std::move(o));
  };
}
#endif

void negative_lambda_checks() {
  auto never_moves_nested = [](Obj&& o1) {
    auto nested = [&]() {
      Obj other{std::move(o1)};
    };
  };

#if __cplusplus >= 201402L
  auto auto_lvalue_ref_param = [](auto& o1) {
    Obj other{o1};
  };

  auto auto_forwarding_ref_param = [](auto&& o1) {
    Obj other{o1};
  };

  auto does_move_auto_rvalue_ref_param = [](auto&& o1) {
    Obj other{std::forward(o1)};
  };
#endif

  auto does_move = [](Obj&& o1) {
    Obj other{std::move(o1)};
  };

  auto not_rvalue_ref = [](Obj& o1) {
    Obj other{std::move(o1)};
  };

  Obj local;
  auto captures = [local]() { };
}

struct AClass {
  void member_with_lambda_no_move(Obj&& o) {
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: rvalue reference parameter 'o' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
    auto captures_this = [=, this]() {
      Obj other = std::move(o);
    };
  }
  void member_with_lambda_that_moves(Obj&& o) {
    auto captures_this = [&, this]() {
      Obj other = std::move(o);
    };
  }
};

void useless_move(Obj&& o) {
  // FIXME - The object is not actually moved from - this should probably be
  // flagged by *some* check. Which one?
  std::move(o);
}

template <typename>
class TemplatedClass;

template <typename T>
void unresolved_lookup(TemplatedClass<T>&& o) {
  TemplatedClass<T> moved = std::move(o);
}

struct DefinesMove {
  DefinesMove(DefinesMove&& rhs) : o(std::move(rhs.o)) { }
  DefinesMove& operator=(DefinesMove&& rhs) {
    if (this != &rhs) {
      o = std::move(rhs.o);
    }
    return *this;
  }
  Obj o;
};

struct DeclaresMove {
  DeclaresMove(DeclaresMove&& rhs);
  DeclaresMove& operator=(DeclaresMove&& rhs);
};

struct AnotherObj {
  AnotherObj(Obj&& o) : o(std::move(o)) {}
  AnotherObj(Obj&& o, int) { o = std::move(o); }
  Obj o;
};

template <class T>
struct AClassTemplate {
  AClassTemplate(T&& t) {}
  // CHECK-MESSAGES-NONDEDUCED: :[[@LINE-1]]:22: warning: rvalue reference parameter 't' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]

  void moves(T&& t) {
    T other = std::move(t);
  }
  void never_moves(T&& t) {}
  // CHECK-MESSAGES-NONDEDUCED: :[[@LINE-1]]:24: warning: rvalue reference parameter 't' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
};

void instantiate_a_class_template() {
  Obj o;
  AClassTemplate<Obj> withObj{std::move(o)};
  withObj.never_moves(std::move(o));

  AClassTemplate<Obj&> withObjRef(o);
  withObjRef.never_moves(o);
}

namespace gh68209
{
  void f1([[maybe_unused]] int&& x) {}

  void f2(__attribute__((unused)) int&& x) {}

  void f3(int&& x) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: rvalue reference parameter 'x' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]

  template <typename T>
  void f4([[maybe_unused]] T&& x) {}

  template <typename T>
  void f5(__attribute((unused)) T&& x) {}

  template<typename T>
  void f6(T&& x) {}

  void f7([[maybe_unused]] int&& x) { x += 1; }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: rvalue reference parameter 'x' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]

  void f8(__attribute__((unused)) int&& x) { x += 1; }
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: rvalue reference parameter 'x' is never moved from inside the function body [cppcoreguidelines-rvalue-reference-param-not-moved]
} // namespace gh68209
