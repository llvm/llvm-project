// RUN: %check_clang_tidy %s bugprone-stringview-nullptr -std=c++17 %t

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
  basic_string_view();

  basic_string_view(const CharT *);

  // Not present in C++17 and C++20:
  // basic_string_view(std::nullptr_t);

  basic_string_view(const CharT *, size_t);

  basic_string_view(const basic_string_view &);

  basic_string_view &operator=(const basic_string_view &);
};

template <typename CharT>
bool operator<(basic_string_view<CharT>, basic_string_view<CharT>);
template <typename CharT>
bool operator<(type_identity_t<basic_string_view<CharT>>,
               basic_string_view<CharT>);
template <typename CharT>
bool operator<(basic_string_view<CharT>,
               type_identity_t<basic_string_view<CharT>>);

template <typename CharT>
bool operator<=(basic_string_view<CharT>, basic_string_view<CharT>);
template <typename CharT>
bool operator<=(type_identity_t<basic_string_view<CharT>>,
                basic_string_view<CharT>);
template <typename CharT>
bool operator<=(basic_string_view<CharT>,
                type_identity_t<basic_string_view<CharT>>);

template <typename CharT>
bool operator>(basic_string_view<CharT>, basic_string_view<CharT>);
template <typename CharT>
bool operator>(type_identity_t<basic_string_view<CharT>>,
               basic_string_view<CharT>);
template <typename CharT>
bool operator>(basic_string_view<CharT>,
               type_identity_t<basic_string_view<CharT>>);

template <typename CharT>
bool operator>=(basic_string_view<CharT>, basic_string_view<CharT>);
template <typename CharT>
bool operator>=(type_identity_t<basic_string_view<CharT>>,
                basic_string_view<CharT>);
template <typename CharT>
bool operator>=(basic_string_view<CharT>,
                type_identity_t<basic_string_view<CharT>>);

template <typename CharT>
bool operator==(basic_string_view<CharT>, basic_string_view<CharT>);
template <typename CharT>
bool operator==(type_identity_t<basic_string_view<CharT>>,
                basic_string_view<CharT>);
template <typename CharT>
bool operator==(basic_string_view<CharT>,
                type_identity_t<basic_string_view<CharT>>);

template <typename CharT>
bool operator!=(basic_string_view<CharT>, basic_string_view<CharT>);
template <typename CharT>
bool operator!=(type_identity_t<basic_string_view<CharT>>,
                basic_string_view<CharT>);
template <typename CharT>
bool operator!=(basic_string_view<CharT>,
                type_identity_t<basic_string_view<CharT>>);

using string_view = basic_string_view<char>;

} // namespace std

void function(std::string_view);
void function(std::string_view, std::string_view);

void temporary_construction() /* a */ {
  // Functional Cast
  {
    (void)(std::string_view(nullptr)) /* a1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing basic_string_view from null is undefined; replace with the default constructor
    // CHECK-FIXES: {{^}}    (void)(std::string_view()) /* a1 */;

    (void)(std::string_view((nullptr))) /* a2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view()) /* a2 */;

    (void)(std::string_view({nullptr})) /* a3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view()) /* a3 */;

    (void)(std::string_view({(nullptr)})) /* a4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view()) /* a4 */;

    (void)(std::string_view({})) /* a5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view()) /* a5 */;
  }

  // Temporary Object
  {
    (void)(std::string_view{nullptr}) /* a6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view{}) /* a6 */;

    (void)(std::string_view{(nullptr)}) /* a7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view{}) /* a7 */;

    (void)(std::string_view{{nullptr}}) /* a8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view{}) /* a8 */;

    (void)(std::string_view{{(nullptr)}}) /* a9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view{}) /* a9 */;

    (void)(std::string_view{{}}) /* a10 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(std::string_view{}) /* a10 */;
  }

  // C-Style Cast && Compound Literal
  {
    (void)((std::string_view) nullptr) /* a11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view) {}) /* a11 */;

    (void)((std::string_view)(nullptr)) /* a12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view){}) /* a12 */;

    (void)((std::string_view){nullptr}) /* a13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view){}) /* a13 */;

    (void)((std::string_view){(nullptr)}) /* a14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view){}) /* a14 */;

    (void)((std::string_view){{nullptr}}) /* a15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view){}) /* a15 */;

    (void)((std::string_view){{(nullptr)}}) /* a16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view){}) /* a16 */;

    (void)((std::string_view){{}}) /* a17 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((std::string_view){}) /* a17 */;

    (void)((const std::string_view) nullptr) /* a18 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view) {}) /* a18 */;

    (void)((const std::string_view)(nullptr)) /* a19 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view){}) /* a19 */;

    (void)((const std::string_view){nullptr}) /* a20 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view){}) /* a20 */;

    (void)((const std::string_view){(nullptr)}) /* a21 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view){}) /* a21 */;

    (void)((const std::string_view){{nullptr}}) /* a22 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view){}) /* a22 */;

    (void)((const std::string_view){{(nullptr)}}) /* a23 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view){}) /* a23 */;

    (void)((const std::string_view){{}}) /* a24 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)((const std::string_view){}) /* a24 */;
  }

  // Static Cast
  {
    (void)(static_cast<std::string_view>(nullptr)) /* a25 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(static_cast<std::string_view>("")) /* a25 */;

    (void)(static_cast<std::string_view>((nullptr))) /* a26 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(static_cast<std::string_view>("")) /* a26 */;

    (void)(static_cast<const std::string_view>(nullptr)) /* a27 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(static_cast<const std::string_view>("")) /* a27 */;

    (void)(static_cast<const std::string_view>((nullptr))) /* a28 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(static_cast<const std::string_view>("")) /* a28 */;
  }
}

void stack_construction() /* b */ {
  // Copy Initialization
  {
    std::string_view b1 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b1 = {};

    std::string_view b2 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b2 = {};

    const std::string_view b3 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b3 = {};

    const std::string_view b4 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b4 = {};
  }

  // Copy List Initialization
  {
    std::string_view b5 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b5 = {};

    std::string_view b6 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b6 = {};

    std::string_view b7 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b7 = {};

    std::string_view b8 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b8 = {};

    std::string_view b9 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b9 = {};

    const std::string_view b10 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b10 = {};

    const std::string_view b11 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b11 = {};

    const std::string_view b12 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b12 = {};

    const std::string_view b13 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b13 = {};

    const std::string_view b14 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b14 = {};
  }

  // Direct Initialization
  {
    std::string_view b15(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b15;

    std::string_view b16((nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b16;

    std::string_view b17({nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b17;

    std::string_view b18({(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b18;

    std::string_view b19({}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b19;

    const std::string_view b20(nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b20;

    const std::string_view b21((nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b21;

    const std::string_view b22({nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b22;

    const std::string_view b23({(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b23;

    const std::string_view b24({}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b24;
  }

  // Direct List Initialization
  {
    std::string_view b25{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b25{};

    std::string_view b26{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b26{};

    std::string_view b27{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b27{};

    std::string_view b28{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b28{};

    std::string_view b29{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view b29{};

    const std::string_view b30{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b30{};

    const std::string_view b31{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b31{};

    const std::string_view b32{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b32{};

    const std::string_view b33{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b33{};

    const std::string_view b34{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view b34{};
  }
}

void field_construction() /* c */ {
  struct DefaultMemberInitializers {
    void CopyInitialization();

    std::string_view c1 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c1 = {};

    std::string_view c2 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c2 = {};

    const std::string_view c3 = nullptr;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c3 = {};

    const std::string_view c4 = (nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c4 = {};

    void CopyListInitialization();

    std::string_view c5 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c5 = {};

    std::string_view c6 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c6 = {};

    std::string_view c7 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c7 = {};

    std::string_view c8 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c8 = {};

    std::string_view c9 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c9 = {};

    const std::string_view c10 = {nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c10 = {};

    const std::string_view c11 = {(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c11 = {};

    const std::string_view c12 = {{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c12 = {};

    const std::string_view c13 = {{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c13 = {};

    const std::string_view c14 = {{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c14 = {};

    void DirectListInitialization();

    std::string_view c15{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c15{};

    std::string_view c16{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c16{};

    std::string_view c17{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c17{};

    std::string_view c18{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c18{};

    std::string_view c19{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    std::string_view c19{};

    const std::string_view c20{nullptr};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c20{};

    const std::string_view c21{(nullptr)};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c21{};

    const std::string_view c22{{nullptr}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c22{};

    const std::string_view c23{{(nullptr)}};
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c23{};

    const std::string_view c24{{}}; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    const std::string_view c24{};
  };

  class ConstructorInitializers {
    ConstructorInitializers()
        : direct_initialization(),

          c25(nullptr),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c25(),

          c26((nullptr)),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c26(),

          c27({nullptr}),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c27(),

          c28({(nullptr)}),
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c28(),

          c29({}), // Default `const CharT*`
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c29(),

          direct_list_initialization(),

          c30{nullptr},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c30{},

          c31{(nullptr)},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c31{},

          c32{{nullptr}},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c32{},

          c33{{(nullptr)}},
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c33{},

          c34{{}}, // Default `const CharT*`
          // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
          // CHECK-FIXES: {{^}}          c34{},

          end_of_list() {}

    std::nullptr_t direct_initialization;
    std::string_view c25;
    std::string_view c26;
    std::string_view c27;
    std::string_view c28;
    std::string_view c29;
    std::nullptr_t direct_list_initialization;
    std::string_view c30;
    std::string_view c31;
    std::string_view c32;
    std::string_view c33;
    std::string_view c34;
    std::nullptr_t end_of_list;
  };
}

void default_argument_construction() /* d */ {
  // Copy Initialization
  {
    void d1(std::string_view sv = nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d1(std::string_view sv = {});

    void d2(std::string_view sv = (nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d2(std::string_view sv = {});

    void d3(const std::string_view sv = nullptr);
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d3(const std::string_view sv = {});

    void d4(const std::string_view sv = (nullptr));
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d4(const std::string_view sv = {});
  }

  // Copy List Initialization
  {
    void d5(std::string_view sv = {nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d5(std::string_view sv = {});

    void d6(std::string_view sv = {(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d6(std::string_view sv = {});

    void d7(std::string_view sv = {{nullptr}});
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d7(std::string_view sv = {});

    void d8(std::string_view sv = {{(nullptr)}});
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d8(std::string_view sv = {});

    void d9(std::string_view sv = {{}}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d9(std::string_view sv = {});

    void d10(const std::string_view sv = {nullptr});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d10(const std::string_view sv = {});

    void d11(const std::string_view sv = {(nullptr)});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d11(const std::string_view sv = {});

    void d12(const std::string_view sv = {{nullptr}});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d12(const std::string_view sv = {});

    void d13(const std::string_view sv = {{(nullptr)}});
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d13(const std::string_view sv = {});

    void d14(const std::string_view sv = {{}}); // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    void d14(const std::string_view sv = {});
  }
}

void heap_construction() /* e */ {
  // Direct Initialization
  {
    (void)(new std::string_view(nullptr)) /* e1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view()) /* e1 */;

    (void)(new std::string_view((nullptr))) /* e2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view()) /* e2 */;

    (void)(new std::string_view({nullptr})) /* e3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view()) /* e3 */;

    (void)(new std::string_view({(nullptr)})) /* e4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view()) /* e4 */;

    (void)(new std::string_view({})) /* e5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view()) /* e5 */;

    (void)(new const std::string_view(nullptr)) /* e6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view()) /* e6 */;

    (void)(new const std::string_view((nullptr))) /* e7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view()) /* e7 */;

    (void)(new const std::string_view({nullptr})) /* e8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view()) /* e8 */;

    (void)(new const std::string_view({(nullptr)})) /* e9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view()) /* e9 */;

    (void)(new const std::string_view({})) /* e10 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view()) /* e10 */;
  }

  // Direct List Initialization
  {
    (void)(new std::string_view{nullptr}) /* e11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view{}) /* e11 */;

    (void)(new std::string_view{(nullptr)}) /* e12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view{}) /* e12 */;

    (void)(new std::string_view{{nullptr}}) /* e13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view{}) /* e13 */;

    (void)(new std::string_view{{(nullptr)}}) /* e14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view{}) /* e14 */;

    (void)(new std::string_view{{}}) /* e15 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new std::string_view{}) /* e15 */;

    (void)(new const std::string_view{nullptr}) /* e16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view{}) /* e16 */;

    (void)(new const std::string_view{(nullptr)}) /* e17 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view{}) /* e17 */;

    (void)(new const std::string_view{{nullptr}}) /* e18 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view{}) /* e18 */;

    (void)(new const std::string_view{{(nullptr)}}) /* e19 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view{}) /* e19 */;

    (void)(new const std::string_view{{}}) /* e20 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    (void)(new const std::string_view{}) /* e20 */;
  }
}

void function_invocation() /* f */ {
  // Single Argument
  {
    function(nullptr) /* f1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}) /* f1 */;

    function((nullptr)) /* f2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}) /* f2 */;

    function({nullptr}) /* f3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}) /* f3 */;

    function({(nullptr)}) /* f4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}) /* f4 */;

    function({{}}) /* f5 */; // Default `const CharT*`
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}) /* f5 */;
  }

  // Multiple Argument
  {
    function(nullptr, nullptr) /* f6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: constructing{{.*}}default
    // CHECK-MESSAGES: :[[@LINE-2]]:23: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}, {}) /* f6 */;

    function((nullptr), (nullptr)) /* f7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: constructing{{.*}}default
    // CHECK-MESSAGES: :[[@LINE-2]]:25: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}, {}) /* f7 */;

    function({nullptr}, {nullptr}) /* f8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
    // CHECK-MESSAGES: :[[@LINE-2]]:26: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}, {}) /* f8 */;

    function({(nullptr)}, {(nullptr)}) /* f9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
    // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}, {}) /* f9 */;

    function({{}}, {{}}) /* f10 */; // Default `const CharT*`s
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: constructing{{.*}}default
    // CHECK-MESSAGES: :[[@LINE-2]]:21: warning: constructing{{.*}}default
    // CHECK-FIXES: {{^}}    function({}, {}) /* f10 */;
  }
}

void assignment(std::string_view sv) /* g */ {
  sv = nullptr /* g1 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: assignment to basic_string_view from null is undefined; replace with the default constructor
  // CHECK-FIXES: {{^}}  sv = {} /* g1 */;

  sv = (nullptr) /* g2 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  sv = {} /* g2 */;

  sv = {nullptr} /* g3 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  sv = {} /* g3 */;

  sv = {(nullptr)} /* g4 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  sv = {} /* g4 */;

  sv = {{}} /* g5 */; // Default `const CharT*`
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  sv = {} /* g5 */;
}

void pointer_assignment(std::string_view *sv_ptr) /* h */ {
  *sv_ptr = nullptr /* h1 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  *sv_ptr = {} /* h1 */;

  *sv_ptr = (nullptr) /* h2 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  *sv_ptr = {} /* h2 */;

  *sv_ptr = {nullptr} /* h3 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  *sv_ptr = {} /* h3 */;

  *sv_ptr = {(nullptr)} /* h4 */;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  *sv_ptr = {} /* h4 */;

  *sv_ptr = {{}} /* h5 */; // Default `const CharT*`
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: assignment{{.*}}default
  // CHECK-FIXES: {{^}}  *sv_ptr = {} /* h5 */;
}

void lesser_comparison(std::string_view sv) /* i */ {
  // Without Equality
  {
    (void)(sv < nullptr) /* i1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing basic_string_view to null is undefined; replace with the empty string
    // CHECK-FIXES: {{^}}    (void)(sv < "") /* i1 */;

    (void)(sv < (nullptr)) /* i2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv < "") /* i2 */;

    (void)(nullptr < sv) /* i3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" < sv) /* i3 */;

    (void)((nullptr) < sv) /* i4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" < sv) /* i4 */;
  }

  // With Equality
  {
    (void)(sv <= nullptr) /* i5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv <= "") /* i5 */;

    (void)(sv <= (nullptr)) /* i6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv <= "") /* i6 */;

    (void)(nullptr <= sv) /* i7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" <= sv) /* i7 */;

    (void)((nullptr) <= sv) /* i8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" <= sv) /* i8 */;
  }
}

void pointer_lesser_comparison(std::string_view *sv_ptr) /* j */ {
  // Without Equality
  {
    (void)(*sv_ptr < nullptr) /* j1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr < "") /* j1 */;

    (void)(*sv_ptr < (nullptr)) /* j2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr < "") /* j2 */;

    (void)(nullptr < *sv_ptr) /* j3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" < *sv_ptr) /* j3 */;

    (void)((nullptr) < *sv_ptr) /* j4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" < *sv_ptr) /* j4 */;
  }

  // With Equality
  {
    (void)(*sv_ptr <= nullptr) /* j5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr <= "") /* j5 */;

    (void)(*sv_ptr <= (nullptr)) /* j6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr <= "") /* j6 */;

    (void)(nullptr <= *sv_ptr) /* j7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" <= *sv_ptr) /* j7 */;

    (void)((nullptr) <= *sv_ptr) /* j8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" <= *sv_ptr) /* j8 */;
  }
}

void greater_comparison(std::string_view sv) /* k */ {
  // Without Equality
  {
    (void)(sv > nullptr) /* k1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv > "") /* k1 */;

    (void)(sv > (nullptr)) /* k2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv > "") /* k2 */;

    (void)(nullptr > sv) /* k3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" > sv) /* k3 */;

    (void)((nullptr) > sv) /* k4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" > sv) /* k4 */;
  }

  // With Equality
  {
    (void)(sv >= nullptr) /* k5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv >= "") /* k5 */;

    (void)(sv >= (nullptr)) /* k6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(sv >= "") /* k6 */;

    (void)(nullptr >= sv) /* k7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" >= sv) /* k7 */;

    (void)((nullptr) >= sv) /* k8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" >= sv) /* k8 */;
  }
}

void pointer_greater_comparison(std::string_view *sv_ptr) /* l */ {
  // Without Equality
  {
    (void)(*sv_ptr > nullptr) /* l1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr > "") /* l1 */;

    (void)(*sv_ptr > (nullptr)) /* l2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr > "") /* l2 */;

    (void)(nullptr > *sv_ptr) /* l3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" > *sv_ptr) /* l3 */;

    (void)((nullptr) > *sv_ptr) /* l4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" > *sv_ptr) /* l4 */;
  }

  // With Equality
  {
    (void)(*sv_ptr >= nullptr) /* l5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr >= "") /* l5 */;

    (void)(*sv_ptr >= (nullptr)) /* l6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)(*sv_ptr >= "") /* l6 */;

    (void)(nullptr >= *sv_ptr) /* l7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" >= *sv_ptr) /* l7 */;

    (void)((nullptr) >= *sv_ptr) /* l8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}empty string
    // CHECK-FIXES: {{^}}    (void)("" >= *sv_ptr) /* l8 */;
  }
}

void equality_comparison(std::string_view sv) /* m */ {
  // Empty Without Parens
  {
    (void)(sv == nullptr) /* m1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing basic_string_view to null is undefined; replace with the emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m1 */;

    (void)(sv == (nullptr)) /* m2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m2 */;

    (void)(nullptr == sv) /* m3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m3 */;

    (void)((nullptr) == sv) /* m4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m4 */;
  }

  // Empty With Parens
  {
    (void)((sv) == nullptr) /* m5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing basic_string_view to null is undefined; replace with the emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m5 */;

    (void)((sv) == (nullptr)) /* m6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m6 */;

    (void)(nullptr == (sv)) /* m7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m7 */;

    (void)((nullptr) == (sv)) /* m8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv.empty()) /* m8 */;
  }

  // Non-Empty Without Parens
  {
    (void)((sv) != nullptr) /* m9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m9 */;

    (void)((sv) != (nullptr)) /* m10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m10 */;

    (void)(nullptr != (sv)) /* m11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m11 */;

    (void)((nullptr) != (sv)) /* m12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m12 */;
  }

  // Non-Empty With Parens
  {
    (void)((sv) != nullptr) /* m13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m13 */;

    (void)((sv) != (nullptr)) /* m14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m14 */;

    (void)(nullptr != (sv)) /* m15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m15 */;

    (void)((nullptr) != (sv)) /* m16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv.empty()) /* m16 */;
  }
}

void pointer_equality_comparison(std::string_view *sv_ptr) /* n */ {
  // Empty Without Parens
  {
    (void)(*sv_ptr == nullptr) /* n1 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n1 */;

    (void)(*sv_ptr == (nullptr)) /* n2 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n2 */;

    (void)(nullptr == *sv_ptr) /* n3 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n3 */;

    (void)((nullptr) == *sv_ptr) /* n4 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n4 */;
  }

  // Empty With Parens
  {
    (void)((*sv_ptr) == nullptr) /* n5 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n5 */;

    (void)((*sv_ptr) == (nullptr)) /* n6 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n6 */;

    (void)(nullptr == (*sv_ptr)) /* n7 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n7 */;

    (void)((nullptr) == (*sv_ptr)) /* n8 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(sv_ptr->empty()) /* n8 */;
  }

  // Non-Empty With Parens
  {
    (void)((*sv_ptr) != nullptr) /* n9 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n9 */;

    (void)((*sv_ptr) != (nullptr)) /* n10 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n10 */;

    (void)(nullptr != (*sv_ptr)) /* n11 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n11 */;

    (void)((nullptr) != (*sv_ptr)) /* n12 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n12 */;
  }

  // Non-Empty Without Parens
  {
    (void)((*sv_ptr) != nullptr) /* n13 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n13 */;

    (void)((*sv_ptr) != (nullptr)) /* n14 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n14 */;

    (void)(nullptr != (*sv_ptr)) /* n15 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n15 */;

    (void)((nullptr) != (*sv_ptr)) /* n16 */;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: comparing{{.*}}emptiness query
    // CHECK-FIXES: {{^}}    (void)(!sv_ptr->empty()) /* n16 */;
  }
}
