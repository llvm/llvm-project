// RUN: %check_clang_tidy --match-partial-fixes %s bugprone-stringview-nullptr -std=c++17 %t

namespace std {

using size_t = long long;
using nullptr_t = decltype(nullptr);

template <typename T>
T &&declval();

template <typename T>
struct type_identity { using type = T; };
template <typename T>
using type_identity_t = typename type_identity<T>::type;

template <typename CharT>
class basic_string_view {
public:
  constexpr basic_string_view() {}

  constexpr basic_string_view(const CharT *) {}

  // Not present in C++17 and C++20:
  // constexpr basic_string_view(std::nullptr_t) {}

  constexpr basic_string_view(const CharT *, size_t) {}

  constexpr basic_string_view(const basic_string_view &) {}

  constexpr basic_string_view &operator=(const basic_string_view &) { return *this; }
};

template <typename CharT>
constexpr bool operator<(basic_string_view<CharT>, basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator<(type_identity_t<basic_string_view<CharT>>,
                         basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator<(basic_string_view<CharT>,
                         type_identity_t<basic_string_view<CharT>>) {
  return {};
}

template <typename CharT>
constexpr bool operator<=(basic_string_view<CharT>, basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator<=(type_identity_t<basic_string_view<CharT>>,
                          basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator<=(basic_string_view<CharT>,
                          type_identity_t<basic_string_view<CharT>>) {
  return {};
}

template <typename CharT>
constexpr bool operator>(basic_string_view<CharT>, basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator>(type_identity_t<basic_string_view<CharT>>,
                         basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator>(basic_string_view<CharT>,
                         type_identity_t<basic_string_view<CharT>>) {
  return {};
}

template <typename CharT>
constexpr bool operator>=(basic_string_view<CharT>, basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator>=(type_identity_t<basic_string_view<CharT>>,
                          basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator>=(basic_string_view<CharT>,
                          type_identity_t<basic_string_view<CharT>>) {
  return {};
}

template <typename CharT>
constexpr bool operator==(basic_string_view<CharT>, basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator==(type_identity_t<basic_string_view<CharT>>,
                          basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator==(basic_string_view<CharT>,
                          type_identity_t<basic_string_view<CharT>>) {
  return {};
}

template <typename CharT>
constexpr bool operator!=(basic_string_view<CharT>, basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator!=(type_identity_t<basic_string_view<CharT>>,
                          basic_string_view<CharT>) {
  return {};
}
template <typename CharT>
constexpr bool operator!=(basic_string_view<CharT>,
                          type_identity_t<basic_string_view<CharT>>) {
  return {};
}

using string_view = basic_string_view<char>;

} // namespace std

using SV = std::string_view; // Used in some places for shorter line length

void function(std::string_view);
void function(std::string_view, std::string_view);

void temporary_construction() /* a */ {
  // Functional Cast
  {
    (void)(std::string_view(nullptr)) /* a1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing basic_string_view from null is undefined; replace with the default constructor
    // CHECK-FIXES: (void)(std::string_view()) /* a1 */;

    (void)(std::string_view((nullptr))) /* a2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view()) /* a2 */;

    (void)(std::string_view({nullptr})) /* a3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view()) /* a3 */;

    (void)(std::string_view({(nullptr)})) /* a4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view()) /* a4 */;

    (void)(std::string_view({})) /* a5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view()) /* a5 */;
  }

  // Temporary Object
  {
    (void)(std::string_view{nullptr}) /* a6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view{}) /* a6 */;

    (void)(std::string_view{(nullptr)}) /* a7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view{}) /* a7 */;

    (void)(std::string_view{{nullptr}}) /* a8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view{}) /* a8 */;

    (void)(std::string_view{{(nullptr)}}) /* a9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view{}) /* a9 */;

    (void)(std::string_view{{}}) /* a10 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(std::string_view{}) /* a10 */;
  }

  // C-Style Cast && Compound Literal
  {
    (void)((std::string_view) nullptr) /* a11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view) {}) /* a11 */;

    (void)((std::string_view)(nullptr)) /* a12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view){}) /* a12 */;

    (void)((std::string_view){nullptr}) /* a13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view){}) /* a13 */;

    (void)((std::string_view){(nullptr)}) /* a14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view){}) /* a14 */;

    (void)((std::string_view){{nullptr}}) /* a15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view){}) /* a15 */;

    (void)((std::string_view){{(nullptr)}}) /* a16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view){}) /* a16 */;

    (void)((std::string_view){{}}) /* a17 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((std::string_view){}) /* a17 */;

    (void)((const std::string_view) nullptr) /* a18 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view) {}) /* a18 */;

    (void)((const std::string_view)(nullptr)) /* a19 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view){}) /* a19 */;

    (void)((const std::string_view){nullptr}) /* a20 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view){}) /* a20 */;

    (void)((const std::string_view){(nullptr)}) /* a21 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view){}) /* a21 */;

    (void)((const std::string_view){{nullptr}}) /* a22 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view){}) /* a22 */;

    (void)((const std::string_view){{(nullptr)}}) /* a23 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view){}) /* a23 */;

    (void)((const std::string_view){{}}) /* a24 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)((const std::string_view){}) /* a24 */;
  }

  // Static Cast
  {
    (void)(static_cast<std::string_view>(nullptr)) /* a25 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: casting to basic_string_view from null is undefined; replace with the empty string
    // CHECK-FIXES: (void)(static_cast<std::string_view>("")) /* a25 */;

    (void)(static_cast<std::string_view>((nullptr))) /* a26 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: casting{{.*}}empty string
    // CHECK-FIXES: (void)(static_cast<std::string_view>("")) /* a26 */;

    (void)(static_cast<const std::string_view>(nullptr)) /* a27 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: casting{{.*}}empty string
    // CHECK-FIXES: (void)(static_cast<const std::string_view>("")) /* a27 */;

    (void)(static_cast<const std::string_view>((nullptr))) /* a28 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: casting{{.*}}empty string
    // CHECK-FIXES: (void)(static_cast<const std::string_view>("")) /* a28 */;
  }
}

void stack_construction() /* b */ {
  // Copy Initialization
  {
    std::string_view b1 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b1 = {};

    std::string_view b2 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b2 = {};

    const std::string_view b3 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b3 = {};

    const std::string_view b4 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b4 = {};
  }

  // Copy Initialization With Temporary
  {
    std::string_view b5 = std::string_view(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b5 = std::string_view();

    std::string_view b6 = std::string_view{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b6 = std::string_view{};

    std::string_view b7 = (std::string_view) nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b7 = (std::string_view) {};

    std::string_view b8 = (std::string_view){nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b8 = (std::string_view){};

    std::string_view b9 = static_cast<SV>(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view b9 = static_cast<SV>("");
  }

  // Copy List Initialization
  {
    std::string_view b10 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b10 = {};

    std::string_view b11 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b11 = {};

    std::string_view b12 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b12 = {};

    std::string_view b13 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b13 = {};

    std::string_view b14 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b14 = {};

    const std::string_view b15 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b15 = {};

    const std::string_view b16 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b16 = {};

    const std::string_view b17 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b17 = {};

    const std::string_view b18 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b18 = {};

    const std::string_view b19 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b19 = {};
  }

  // Copy List Initialization With Temporary
  {
    std::string_view b20 = {std::string_view(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b20 = {std::string_view()};

    std::string_view b21 = {std::string_view{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b21 = {std::string_view{}};

    std::string_view b22 = {(std::string_view) nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b22 = {(std::string_view) {}};

    std::string_view b23 = {(std::string_view){nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b23 = {(std::string_view){}};

    std::string_view b24 = {static_cast<SV>(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view b24 = {static_cast<SV>("")};
  }

  // Direct Initialization
  {
    std::string_view b25(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b25;

    std::string_view b26((nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b26;

    std::string_view b27({nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b27;

    std::string_view b28({(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b28;

    std::string_view b29({}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b29;

    const std::string_view b30(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b30;

    const std::string_view b31((nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b31;

    const std::string_view b32({nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b32;

    const std::string_view b33({(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b33;

    const std::string_view b34({}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b34;
  }

  // Direct Initialization With Temporary
  {
    std::string_view b35(std::string_view(nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b35(std::string_view());

    std::string_view b36(std::string_view{nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b36(std::string_view{});

    std::string_view b37((std::string_view) nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b37((std::string_view) {});

    std::string_view b38((std::string_view){nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b38((std::string_view){});

    std::string_view b39(static_cast<SV>(nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view b39(static_cast<SV>(""));
  }

  // Direct List Initialization
  {
    std::string_view b40{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b40{};

    std::string_view b41{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b41{};

    std::string_view b42{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b42{};

    std::string_view b43{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b43{};

    std::string_view b44{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b44{};

    const std::string_view b45{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b45{};

    const std::string_view b46{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b46{};

    const std::string_view b47{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b47{};

    const std::string_view b48{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b48{};

    const std::string_view b49{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view b49{};
  }

  // Direct List Initialization With Temporary
  {
    std::string_view b50{std::string_view(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b50{std::string_view()};

    std::string_view b51{std::string_view{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b51{std::string_view{}};

    std::string_view b52{(std::string_view) nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b52{(std::string_view) {}};

    std::string_view b53{(std::string_view){nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view b53{(std::string_view){}};

    std::string_view b54{static_cast<SV>(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view b54{static_cast<SV>("")};
  }
}

void field_construction() /* c */ {
  // Default Member Initializaers

  struct DMICopyInitialization {
    std::string_view c1 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c1 = {};

    std::string_view c2 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c2 = {};

    const std::string_view c3 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c3 = {};

    const std::string_view c4 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c4 = {};
  };

  struct DMICopyInitializationWithTemporary {
    std::string_view c5 = std::string_view(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c5 = std::string_view();

    std::string_view c6 = std::string_view{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c6 = std::string_view{};

    std::string_view c7 = (std::string_view) nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c7 = (std::string_view) {};

    std::string_view c8 = (std::string_view){nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c8 = (std::string_view){};

    std::string_view c9 = static_cast<SV>(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view c9 = static_cast<SV>("");
  };

  struct DMICopyListInitialization {
    std::string_view c10 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c10 = {};

    std::string_view c11 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c11 = {};

    std::string_view c12 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c12 = {};

    std::string_view c13 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c13 = {};

    std::string_view c14 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c14 = {};

    const std::string_view c15 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c15 = {};

    const std::string_view c16 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c16 = {};

    const std::string_view c17 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c17 = {};

    const std::string_view c18 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c18 = {};

    const std::string_view c19 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c19 = {};
  };

  struct DMICopyListInitializationWithTemporary {
    std::string_view c20 = {std::string_view(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c20 = {std::string_view()};

    std::string_view c21 = {std::string_view{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c21 = {std::string_view{}};

    std::string_view c22 = {(std::string_view) nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c22 = {(std::string_view) {}};

    std::string_view c23 = {(std::string_view){nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c23 = {(std::string_view){}};

    std::string_view c24 = {static_cast<SV>(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view c24 = {static_cast<SV>("")};
  };

  struct DMIDirectListInitialization {
    std::string_view c25{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c25{};

    std::string_view c26{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c26{};

    std::string_view c27{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c27{};

    std::string_view c28{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c28{};

    std::string_view c29{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c29{};

    const std::string_view c30{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c30{};

    const std::string_view c31{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c31{};

    const std::string_view c32{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c32{};

    const std::string_view c33{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c33{};

    const std::string_view c34{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: const std::string_view c34{};
  };

  struct DMIDirectListInitializationWithTemporary {
    std::string_view c35{std::string_view(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c35{std::string_view()};

    std::string_view c36{std::string_view{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c36{std::string_view{}};

    std::string_view c37{(std::string_view) nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c37{(std::string_view) {}};

    std::string_view c38{(std::string_view){nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: constructing{{.*}}default
    // CHECK-FIXES: std::string_view c38{(std::string_view){}};

    std::string_view c39{static_cast<SV>(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: casting{{.*}}empty string
    // CHECK-FIXES: std::string_view c39{static_cast<SV>("")};
  };

  // Constructor Initializers

  class CIDirectInitialization {
    std::string_view c40;
    std::string_view c41;
    std::string_view c42;
    std::string_view c43;
    std::string_view c44;

    CIDirectInitialization()
        : c40(nullptr),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: : c40(),

          c41((nullptr)),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: c41(),

          c42({nullptr}),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: c42(),

          c43({(nullptr)}),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: c43(),

          c44({}) { // Default `const CharT*`
      // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
      // CHECK-FIXES: c44() {
    }
  };

  class CIDirectInitializationWithTemporary {
    std::string_view c45;
    std::string_view c46;
    std::string_view c47;
    std::string_view c48;
    std::string_view c49;

    CIDirectInitializationWithTemporary()
        : c45(std::string_view(nullptr)),
          // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
          // CHECK-FIXES: : c45(std::string_view()),

          c46(std::string_view{nullptr}),
          // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
          // CHECK-FIXES: c46(std::string_view{}),

          c47((std::string_view) nullptr),
          // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
          // CHECK-FIXES: c47((std::string_view) {}),

          c48((std::string_view){nullptr}),
          // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
          // CHECK-FIXES: c48((std::string_view){}),

          c49(static_cast<SV>(nullptr)) {
      // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: casting{{.*}}empty string
      // CHECK-FIXES: c49(static_cast<SV>("")) {
    }
  };

  class CIDirectListInitialization {
    std::string_view c50;
    std::string_view c51;
    std::string_view c52;
    std::string_view c53;
    std::string_view c54;

    CIDirectListInitialization()
        : c50{nullptr},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: : c50{},

          c51{(nullptr)},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: c51{},

          c52{{nullptr}},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: c52{},

          c53{{(nullptr)}},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: c53{},

          c54{{}} { // Default `const CharT*`
      // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
      // CHECK-FIXES: c54{} {
    }
  };

  class CIDirectListInitializationWithTemporary {
    std::string_view c55;
    std::string_view c56;
    std::string_view c57;
    std::string_view c58;
    std::string_view c59;

    CIDirectListInitializationWithTemporary()
        : c55{std::string_view(nullptr)},
          // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
          // CHECK-FIXES: : c55{std::string_view()},

          c56{std::string_view{nullptr}},
          // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
          // CHECK-FIXES: c56{std::string_view{}},

          c57{(std::string_view) nullptr},
          // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
          // CHECK-FIXES: c57{(std::string_view) {}},

          c58{(std::string_view){nullptr}},
          // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
          // CHECK-FIXES: c58{(std::string_view){}},

          c59{static_cast<SV>(nullptr)} {
      // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: casting{{.*}}empty string
      // CHECK-FIXES: c59{static_cast<SV>("")} {
    }
  };
}

void default_argument_construction() /* d */ {
  // Copy Initialization
  {
    void d1(std::string_view sv = nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: void d1(std::string_view sv = {});

    void d2(std::string_view sv = (nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: void d2(std::string_view sv = {});

    void d3(const std::string_view sv = nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: constructing{{.*}}default
    // CHECK-FIXES: void d3(const std::string_view sv = {});

    void d4(const std::string_view sv = (nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: constructing{{.*}}default
    // CHECK-FIXES: void d4(const std::string_view sv = {});
  }

  // Copy Initialization With Temporary
  {
    void d5(std::string_view sv = std::string_view(nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: constructing{{.*}}default
    // CHECK-FIXES: void d5(std::string_view sv = std::string_view());

    void d6(std::string_view sv = std::string_view{nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: constructing{{.*}}default
    // CHECK-FIXES: void d6(std::string_view sv = std::string_view{});

    void d7(std::string_view sv = (std::string_view) nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: constructing{{.*}}default
    // CHECK-FIXES: void d7(std::string_view sv = (std::string_view) {});

    void d8(std::string_view sv = (std::string_view){nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: constructing{{.*}}default
    // CHECK-FIXES: void d8(std::string_view sv = (std::string_view){});

    void d9(std::string_view sv = static_cast<SV>(nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:51: warning: casting{{.*}}empty string
    // CHECK-FIXES: void d9(std::string_view sv = static_cast<SV>(""));
  }

  // Copy List Initialization
  {
    void d10(std::string_view sv = {nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: void d10(std::string_view sv = {});

    void d11(std::string_view sv = {(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: void d11(std::string_view sv = {});

    void d12(std::string_view sv = {{nullptr}});
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: void d12(std::string_view sv = {});

    void d13(std::string_view sv = {{(nullptr)}});
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: void d13(std::string_view sv = {});

    void d14(std::string_view sv = {{}}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: void d14(std::string_view sv = {});

    void d15(const std::string_view sv = {nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: void d15(const std::string_view sv = {});

    void d16(const std::string_view sv = {(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: void d16(const std::string_view sv = {});

    void d17(const std::string_view sv = {{nullptr}});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: void d17(const std::string_view sv = {});

    void d18(const std::string_view sv = {{(nullptr)}});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: void d18(const std::string_view sv = {});

    void d19(const std::string_view sv = {{}}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: void d19(const std::string_view sv = {});
  }

  // Copy List Initialization With Temporary
  {
    void d20(std::string_view sv = {std::string_view(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: constructing{{.*}}default
    // CHECK-FIXES: void d20(std::string_view sv = {std::string_view()});

    void d21(std::string_view sv = {std::string_view{nullptr}});
    // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: constructing{{.*}}default
    // CHECK-FIXES: void d21(std::string_view sv = {std::string_view{}});

    void d22(std::string_view sv = {(std::string_view) nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: constructing{{.*}}default
    // CHECK-FIXES: void d22(std::string_view sv = {(std::string_view) {}});

    void d23(std::string_view sv = {(std::string_view){nullptr}});
    // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: constructing{{.*}}default
    // CHECK-FIXES: void d23(std::string_view sv = {(std::string_view){}});

    void d24(std::string_view sv = {static_cast<SV>(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:53: warning: casting{{.*}}empty string
    // CHECK-FIXES: void d24(std::string_view sv = {static_cast<SV>("")});
  }
}

void heap_construction() /* e */ {
  // Direct Initialization
  {
    (void)(new std::string_view(nullptr)) /* e1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view()) /* e1 */;

    (void)(new std::string_view((nullptr))) /* e2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view()) /* e2 */;

    (void)(new std::string_view({nullptr})) /* e3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view()) /* e3 */;

    (void)(new std::string_view({(nullptr)})) /* e4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view()) /* e4 */;

    (void)(new std::string_view({})) /* e5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view()) /* e5 */;

    (void)(new const std::string_view(nullptr)) /* e6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view()) /* e6 */;

    (void)(new const std::string_view((nullptr))) /* e7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view()) /* e7 */;

    (void)(new const std::string_view({nullptr})) /* e8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view()) /* e8 */;

    (void)(new const std::string_view({(nullptr)})) /* e9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view()) /* e9 */;

    (void)(new const std::string_view({})) /* e10 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view()) /* e10 */;
  }

  // Direct Initialization With Temporary
  {
    (void)(new std::string_view(std::string_view(nullptr))) /* e11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:50: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view(std::string_view())) /* e11 */;

    (void)(new std::string_view(std::string_view{nullptr})) /* e12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:50: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view(std::string_view{})) /* e12 */;

    (void)(new std::string_view((std::string_view) nullptr)) /* e13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view((std::string_view) {})) /* e13 */;

    (void)(new std::string_view((std::string_view){nullptr})) /* e14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view((std::string_view){})) /* e14 */;

    (void)(new std::string_view(static_cast<SV>(nullptr))) /* e15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:49: warning: casting{{.*}}empty string
    // CHECK-FIXES: (void)(new std::string_view(static_cast<SV>(""))) /* e15 */;
  }

  // Direct List Initialization
  {
    (void)(new std::string_view{nullptr}) /* e16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{}) /* e16 */;

    (void)(new std::string_view{(nullptr)}) /* e17 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{}) /* e17 */;

    (void)(new std::string_view{{nullptr}}) /* e18 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{}) /* e18 */;

    (void)(new std::string_view{{(nullptr)}}) /* e19 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{}) /* e19 */;

    (void)(new std::string_view{{}}) /* e20 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{}) /* e20 */;

    (void)(new const std::string_view{nullptr}) /* e21 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view{}) /* e21 */;

    (void)(new const std::string_view{(nullptr)}) /* e22 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view{}) /* e22 */;

    (void)(new const std::string_view{{nullptr}}) /* e23 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view{}) /* e23 */;

    (void)(new const std::string_view{{(nullptr)}}) /* e24 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view{}) /* e24 */;

    (void)(new const std::string_view{{}}) /* e25 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new const std::string_view{}) /* e25 */;
  }

  // Direct List Initialization With Temporary
  {
    (void)(new std::string_view{std::string_view(nullptr)}) /* e26 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:50: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{std::string_view()}) /* e26 */;

    (void)(new std::string_view{std::string_view{nullptr}}) /* e27 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:50: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{std::string_view{}}) /* e27 */;

    (void)(new std::string_view{(std::string_view) nullptr}) /* e28 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{(std::string_view) {}}) /* e28 */;

    (void)(new std::string_view{(std::string_view){nullptr}}) /* e29 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: constructing{{.*}}default
    // CHECK-FIXES: (void)(new std::string_view{(std::string_view){}}) /* e29 */;

    (void)(new std::string_view{static_cast<SV>(nullptr)}) /* e30 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:49: warning: casting{{.*}}empty string
    // CHECK-FIXES: (void)(new std::string_view{static_cast<SV>("")}) /* e30 */;
  }
}

void function_argument_initialization() /* f */ {
  // Function Argument Initialization
  {
    function(nullptr) /* f1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: passing null as basic_string_view is undefined; replace with the empty string
    // CHECK-FIXES: function("") /* f1 */;

    function((nullptr)) /* f2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: passing{{.*}}empty string
    // CHECK-FIXES: function("") /* f2 */;

    function({nullptr}) /* f3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: passing{{.*}}empty string
    // CHECK-FIXES: function("") /* f3 */;

    function({(nullptr)}) /* f4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: passing{{.*}}empty string
    // CHECK-FIXES: function("") /* f4 */;

    function({{}}) /* f5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: passing{{.*}}empty string
    // CHECK-FIXES: function("") /* f5 */;
  }

  // Function Argument Initialization With Temporary
  {
    function(std::string_view(nullptr)) /* f6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: function(std::string_view()) /* f6 */;

    function(std::string_view{nullptr}) /* f7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: function(std::string_view{}) /* f7 */;

    function((std::string_view) nullptr) /* f8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: function((std::string_view) {}) /* f8 */;

    function((std::string_view){nullptr}) /* f9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: function((std::string_view){}) /* f9 */;

    function(static_cast<SV>(nullptr)) /* f10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: casting{{.*}}empty string
    // CHECK-FIXES: function(static_cast<SV>("")) /* f10 */;
  }
}

void assignment(std::string_view sv) /* g */ {
  // Assignment
  {
    sv = nullptr /* g1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: assignment to basic_string_view from null is undefined; replace with the default constructor
    // CHECK-FIXES: sv = {} /* g1 */;

    sv = (nullptr) /* g2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: assignment{{.*}}default
    // CHECK-FIXES: sv = {} /* g2 */;

    sv = {nullptr} /* g3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: assignment{{.*}}default
    // CHECK-FIXES: sv = {} /* g3 */;

    sv = {(nullptr)} /* g4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: assignment{{.*}}default
    // CHECK-FIXES: sv = {} /* g4 */;

    sv = {{}} /* g5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: assignment{{.*}}default
    // CHECK-FIXES: sv = {} /* g5 */;
  }

  // Assignment With Temporary
  {
    sv = std::string_view(nullptr) /* g6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: sv = std::string_view() /* g6 */;

    sv = std::string_view{nullptr} /* g7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: sv = std::string_view{} /* g7 */;

    sv = (std::string_view) nullptr /* g8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: sv = (std::string_view) {} /* g8 */;

    sv = (std::string_view){nullptr} /* g9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: sv = (std::string_view){} /* g9 */;

    sv = static_cast<SV>(nullptr) /* g10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: casting{{.*}}empty string
    // CHECK-FIXES: sv = static_cast<SV>("") /* g10 */;
  }
}

void pointer_assignment(std::string_view *sv_ptr) /* h */ {
  // Assignment
  {
    *sv_ptr = nullptr /* h1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: assignment{{.*}}default
    // CHECK-FIXES: *sv_ptr = {} /* h1 */;

    *sv_ptr = (nullptr) /* h2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: assignment{{.*}}default
    // CHECK-FIXES: *sv_ptr = {} /* h2 */;

    *sv_ptr = {nullptr} /* h3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: assignment{{.*}}default
    // CHECK-FIXES: *sv_ptr = {} /* h3 */;

    *sv_ptr = {(nullptr)} /* h4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: assignment{{.*}}default
    // CHECK-FIXES: *sv_ptr = {} /* h4 */;

    *sv_ptr = {{}} /* h5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: assignment{{.*}}default
    // CHECK-FIXES: *sv_ptr = {} /* h5 */;
  }

  // Assignment With Temporary
  {
    *sv_ptr = std::string_view(nullptr) /* h6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: *sv_ptr = std::string_view() /* h6 */;

    *sv_ptr = std::string_view{nullptr} /* h7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: *sv_ptr = std::string_view{} /* h7 */;

    *sv_ptr = (std::string_view) nullptr /* h8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
    // CHECK-FIXES: *sv_ptr = (std::string_view) {} /* h8 */;

    *sv_ptr = (std::string_view){nullptr} /* h9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
    // CHECK-FIXES: *sv_ptr = (std::string_view){} /* h9 */;

    *sv_ptr = static_cast<SV>(nullptr) /* h10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: casting{{.*}}empty string
    // CHECK-FIXES: *sv_ptr = static_cast<SV>("") /* h10 */;
  }
}

void lesser_comparison(std::string_view sv) /* i */ {
  // Without Equality
  {
    (void)(sv < nullptr) /* i1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing basic_string_view to null is undefined; replace with the empty string
    // CHECK-FIXES: (void)(sv < "") /* i1 */;

    (void)(sv < (nullptr)) /* i2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv < "") /* i2 */;

    (void)(nullptr < sv) /* i3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" < sv) /* i3 */;

    (void)((nullptr) < sv) /* i4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" < sv) /* i4 */;
  }

  // With Equality
  {
    (void)(sv <= nullptr) /* i5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv <= "") /* i5 */;

    (void)(sv <= (nullptr)) /* i6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv <= "") /* i6 */;

    (void)(nullptr <= sv) /* i7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" <= sv) /* i7 */;

    (void)((nullptr) <= sv) /* i8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" <= sv) /* i8 */;
  }
}

void pointer_lesser_comparison(std::string_view *sv_ptr) /* j */ {
  // Without Equality
  {
    (void)(*sv_ptr < nullptr) /* j1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr < "") /* j1 */;

    (void)(*sv_ptr < (nullptr)) /* j2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr < "") /* j2 */;

    (void)(nullptr < *sv_ptr) /* j3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" < *sv_ptr) /* j3 */;

    (void)((nullptr) < *sv_ptr) /* j4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" < *sv_ptr) /* j4 */;
  }

  // With Equality
  {
    (void)(*sv_ptr <= nullptr) /* j5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr <= "") /* j5 */;

    (void)(*sv_ptr <= (nullptr)) /* j6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr <= "") /* j6 */;

    (void)(nullptr <= *sv_ptr) /* j7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" <= *sv_ptr) /* j7 */;

    (void)((nullptr) <= *sv_ptr) /* j8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" <= *sv_ptr) /* j8 */;
  }
}

void greater_comparison(std::string_view sv) /* k */ {
  // Without Equality
  {
    (void)(sv > nullptr) /* k1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv > "") /* k1 */;

    (void)(sv > (nullptr)) /* k2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv > "") /* k2 */;

    (void)(nullptr > sv) /* k3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" > sv) /* k3 */;

    (void)((nullptr) > sv) /* k4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" > sv) /* k4 */;
  }

  // With Equality
  {
    (void)(sv >= nullptr) /* k5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv >= "") /* k5 */;

    (void)(sv >= (nullptr)) /* k6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(sv >= "") /* k6 */;

    (void)(nullptr >= sv) /* k7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" >= sv) /* k7 */;

    (void)((nullptr) >= sv) /* k8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" >= sv) /* k8 */;
  }
}

void pointer_greater_comparison(std::string_view *sv_ptr) /* l */ {
  // Without Equality
  {
    (void)(*sv_ptr > nullptr) /* l1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr > "") /* l1 */;

    (void)(*sv_ptr > (nullptr)) /* l2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr > "") /* l2 */;

    (void)(nullptr > *sv_ptr) /* l3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" > *sv_ptr) /* l3 */;

    (void)((nullptr) > *sv_ptr) /* l4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" > *sv_ptr) /* l4 */;
  }

  // With Equality
  {
    (void)(*sv_ptr >= nullptr) /* l5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr >= "") /* l5 */;

    (void)(*sv_ptr >= (nullptr)) /* l6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)(*sv_ptr >= "") /* l6 */;

    (void)(nullptr >= *sv_ptr) /* l7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" >= *sv_ptr) /* l7 */;

    (void)((nullptr) >= *sv_ptr) /* l8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: (void)("" >= *sv_ptr) /* l8 */;
  }
}

void relative_comparison_with_temporary(std::string_view sv) /* m */ {
  (void)(sv < std::string_view(nullptr)) /* m1 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv < std::string_view()) /* m1 */;

  (void)(sv < std::string_view{nullptr}) /* m2 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv < std::string_view{}) /* m2 */;

  (void)(sv < (std::string_view) nullptr) /* m3 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv < (std::string_view) {}) /* m3 */;

  (void)(sv < (std::string_view){nullptr}) /* m4 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv < (std::string_view){}) /* m4 */;

  (void)(sv < static_cast<SV>(nullptr)) /* m5 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: casting{{.*}}empty string
  // CHECK-FIXES: (void)(sv < static_cast<SV>("")) /* m5 */;
}

void equality_comparison(std::string_view sv) /* n */ {
  // Empty Without Parens
  {
    (void)(sv == nullptr) /* n1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing basic_string_view to null is undefined; replace with the emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n1 */;

    (void)(sv == (nullptr)) /* n2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n2 */;

    (void)(nullptr == sv) /* n3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n3 */;

    (void)((nullptr) == sv) /* n4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n4 */;
  }

  // Empty With Parens
  {
    (void)((sv) == nullptr) /* n5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing basic_string_view to null is undefined; replace with the emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n5 */;

    (void)((sv) == (nullptr)) /* n6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n6 */;

    (void)(nullptr == (sv)) /* n7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n7 */;

    (void)((nullptr) == (sv)) /* n8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv.empty()) /* n8 */;
  }

  // Non-Empty Without Parens
  {
    (void)((sv) != nullptr) /* n9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n9 */;

    (void)((sv) != (nullptr)) /* n10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n10 */;

    (void)(nullptr != (sv)) /* n11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n11 */;

    (void)((nullptr) != (sv)) /* n12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n12 */;
  }

  // Non-Empty With Parens
  {
    (void)((sv) != nullptr) /* n13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n13 */;

    (void)((sv) != (nullptr)) /* n14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n14 */;

    (void)(nullptr != (sv)) /* n15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n15 */;

    (void)((nullptr) != (sv)) /* n16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv.empty()) /* n16 */;
  }
}

void pointer_equality_comparison(std::string_view *sv_ptr) /* o */ {
  // Empty Without Parens
  {
    (void)(*sv_ptr == nullptr) /* o1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o1 */;

    (void)(*sv_ptr == (nullptr)) /* o2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o2 */;

    (void)(nullptr == *sv_ptr) /* o3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o3 */;

    (void)((nullptr) == *sv_ptr) /* o4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o4 */;
  }

  // Empty With Parens
  {
    (void)((*sv_ptr) == nullptr) /* o5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o5 */;

    (void)((*sv_ptr) == (nullptr)) /* o6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o6 */;

    (void)(nullptr == (*sv_ptr)) /* o7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o7 */;

    (void)((nullptr) == (*sv_ptr)) /* o8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(sv_ptr->empty()) /* o8 */;
  }

  // Non-Empty With Parens
  {
    (void)((*sv_ptr) != nullptr) /* o9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o9 */;

    (void)((*sv_ptr) != (nullptr)) /* o10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o10 */;

    (void)(nullptr != (*sv_ptr)) /* o11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o11 */;

    (void)((nullptr) != (*sv_ptr)) /* o12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o12 */;
  }

  // Non-Empty Without Parens
  {
    (void)((*sv_ptr) != nullptr) /* o13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o13 */;

    (void)((*sv_ptr) != (nullptr)) /* o14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o14 */;

    (void)(nullptr != (*sv_ptr)) /* o15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o15 */;

    (void)((nullptr) != (*sv_ptr)) /* o16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: (void)(!sv_ptr->empty()) /* o16 */;
  }
}

void equality_comparison_with_temporary(std::string_view sv) /* p */ {
  (void)(sv == std::string_view(nullptr)) /* p1 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv == std::string_view()) /* p1 */;

  (void)(sv == std::string_view{nullptr}) /* p2 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv == std::string_view{}) /* p2 */;

  (void)(sv == (std::string_view) nullptr) /* p3 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv == (std::string_view) {}) /* p3 */;

  (void)(sv == (std::string_view){nullptr}) /* p4 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
  // CHECK-FIXES: (void)(sv == (std::string_view){}) /* p4 */;

  (void)(sv == static_cast<SV>(nullptr)) /* p5 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: casting{{.*}}empty string
  // CHECK-FIXES: (void)(sv == static_cast<SV>("")) /* p5 */;
}

void return_statement() /* q */ {
  // Return Statement
  {
    []() -> SV { return nullptr; } /* q1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q1 */;

    []() -> SV { return (nullptr); } /* q2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q2 */;

    []() -> SV { return {nullptr}; } /* q3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q3 */;

    []() -> SV { return {(nullptr)}; } /* q4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q4 */;

    []() -> SV { return {{nullptr}}; } /* q5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q5 */;

    []() -> SV { return {{(nullptr)}}; } /* q6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q6 */;

    []() -> SV { return {{}}; } /* q7 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return {}; } /* q7 */;
  }

  // Return Statement With Temporary
  {
    []() -> SV { return SV(nullptr); } /* q8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return SV(); } /* q8 */;

    []() -> SV { return SV{nullptr}; } /* q9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return SV{}; } /* q9 */;

    []() -> SV { return (SV) nullptr; } /* q10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return (SV) {}; } /* q10 */;

    []() -> SV { return (SV){nullptr}; } /* q11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: constructing{{.*}}default
    // CHECK-FIXES: []() -> SV { return (SV){}; } /* q11 */;

    []() -> SV { return static_cast<SV>(nullptr); } /* q12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: casting{{.*}}empty string
    // CHECK-FIXES: []() -> SV { return static_cast<SV>(""); } /* q12 */;
  }
}

void constructor_invocation() /* r */ {
  struct AcceptsSV {
    explicit AcceptsSV(std::string_view) {}
  } r1(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: passing{{.*}}empty string
  // CHECK-FIXES: } r1("");

  (void)(AcceptsSV{nullptr}) /* r2 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: passing{{.*}}empty string
  // CHECK-FIXES: (void)(AcceptsSV{""}) /* r2 */;

  AcceptsSV r3{nullptr};
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: passing{{.*}}empty string
  // CHECK-FIXES: AcceptsSV r3{""};
}
