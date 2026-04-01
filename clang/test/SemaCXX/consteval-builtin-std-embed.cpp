// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-c++2d-extensions -Wno-c++23-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wno-c++2d-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wno-c++2d-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++2d -fsyntax-only --embed-dir=%S/Inputs %s
// expected-no-diagnostics

#depend "resources/**"

#define STR_PREFIX_(a, b) a##b
#define STR_PREFIX(a, b) STR_PREFIX_(a, b)

namespace std {
  enum class byte : unsigned char {};
}

inline constexpr decltype(sizeof(0)) sentinel_size = 44203zu;
template <typename T>
inline constexpr T sentinel_object = {};
template <typename T>
inline constexpr const T* sentinel_address = &sentinel_object<T>;

template <typename T>
struct inspect {
  const T* ptr = sentinel_address<T>;
  decltype(sizeof(0)) size = sentinel_size;
  int status = -1;
};

enum : int {
  not_found,
  found,
  no_depend,
  found_empty
};

template <typename T, typename C>
consteval inspect<T> does_not_exist_core () {
  static constexpr const C name[1] = { C(0) };
  inspect<T> result = {};
  result.ptr = __builtin_std_embed(0b000, result.status, result.size, result.ptr, (sizeof(name) / sizeof(*name))-1, &name[0], 0);
  return result;
}

template <typename T, typename C, decltype(sizeof(0)) N>
consteval inspect<T> non_local_lookup_core (const C (&name)[N]) {
  inspect<T> result = {};
  result.ptr = __builtin_std_embed(0b000, result.status, result.size, result.ptr, (N)-1, &name[0], 0);
  return result;
}

template <typename T, typename C, decltype(sizeof(0)) N>
consteval inspect<T> local_lookup_core (const C (&name)[N]) {
  inspect<T> result = {};
  result.ptr = __builtin_std_embed(0b001, result.status, result.size, result.ptr, (N)-1, &name[0], 0);
  return result;
}

consteval bool does_not_exist () {
  constexpr auto v0 = does_not_exist_core<unsigned char, char>();
  static_assert(v0.status == not_found);
  static_assert(v0.ptr == nullptr);
  static_assert(v0.size == sentinel_size);
  constexpr auto v1 = does_not_exist_core<char, char>();
  static_assert(v1.status == not_found);
  static_assert(v1.ptr == nullptr);
  static_assert(v1.size == sentinel_size);
  constexpr auto v2 = does_not_exist_core<std::byte, char>();
  static_assert(v2.status == not_found);
  static_assert(v2.ptr == nullptr);
  static_assert(v2.size == sentinel_size);

  constexpr auto v3 = does_not_exist_core<unsigned char, wchar_t>();
  static_assert(v3.status == not_found);
  static_assert(v3.ptr == nullptr);
  static_assert(v3.size == sentinel_size);
  constexpr auto v4 = does_not_exist_core<char, wchar_t>();
  static_assert(v4.status == not_found);
  static_assert(v4.ptr == nullptr);
  static_assert(v4.size == sentinel_size);
  constexpr auto v5 = does_not_exist_core<std::byte, wchar_t>();
  static_assert(v5.status == not_found);
  static_assert(v5.ptr == nullptr);
  static_assert(v5.size == sentinel_size);

  constexpr auto v6 = does_not_exist_core<unsigned char, char8_t>();
  static_assert(v6.status == not_found);
  static_assert(v6.ptr == nullptr);
  static_assert(v6.size == sentinel_size);
  constexpr auto v7 = does_not_exist_core<char, char8_t>();
  static_assert(v7.status == not_found);
  static_assert(v7.ptr == nullptr);
  static_assert(v7.size == sentinel_size);
  constexpr auto v8 = does_not_exist_core<std::byte, char8_t>();
  static_assert(v8.status == not_found);
  static_assert(v8.ptr == nullptr);
  static_assert(v8.size == sentinel_size);
  
  return true;
}

consteval bool not_depended_on () {
  constexpr auto v0 = local_lookup_core<unsigned char>( __FILE__);
  static_assert(v0.status == no_depend);
  static_assert(v0.ptr == nullptr);
  static_assert(v0.size == sentinel_size);
  constexpr auto v1 = local_lookup_core<unsigned char>(STR_PREFX(L, __FILE__));
  static_assert(v1.status == no_depend);
  static_assert(v1.ptr == nullptr);
  static_assert(v1.size == sentinel_size);
  constexpr auto v2 = local_lookup_core<unsigned char>(STR_PREFX(u8, __FILE__));
  static_assert(v2.status == no_depend);
  static_assert(v2.ptr == nullptr);
  static_assert(v2.size == sentinel_size);

  constexpr auto v3 = local_lookup_core<char>( __FILE__);
  static_assert(v3.status == no_depend);
  static_assert(v3.ptr == nullptr);
  static_assert(v3.size == sentinel_size);
  constexpr auto v4 = local_lookup_core<char>(STR_PREFX(L, __FILE__));
  static_assert(v4.status == no_depend);
  static_assert(v4.ptr == nullptr);
  static_assert(v4.size == sentinel_size);
  constexpr auto v5 = local_lookup_core<char>(STR_PREFX(u8, __FILE__));
  static_assert(v5.status == no_depend);
  static_assert(v5.ptr == nullptr);
  static_assert(v5.size == sentinel_size);

  constexpr auto v6 = local_lookup_core<std::byte>( __FILE__);
  static_assert(v6.status == no_depend);
  static_assert(v6.ptr == nullptr);
  static_assert(v6.size == sentinel_size);
  constexpr auto v7 = local_lookup_core<std::byte>(STR_PREFX(L, __FILE__));
  static_assert(v7.status == no_depend);
  static_assert(v7.ptr == nullptr);
  static_assert(v7.size == sentinel_size);
  constexpr auto v8 = local_lookup_core<std::byte>(STR_PREFX(u8, __FILE__));
  static_assert(v8.status == no_depend);
  static_assert(v8.ptr == nullptr);
  static_assert(v8.size == sentinel_size);

  return true;
}

consteval bool empty () {
  constexpr auto v0 = non_local_lookup_core<unsigned char>("resources/a/b/empty");
  static_assert(v0.status == found_empty);
  static_assert(v0.ptr == nullptr);
  static_assert(v0.size == 0);
  constexpr auto v1 = non_local_lookup_core<unsigned char>(L"resources/a/b/empty");
  static_assert(v1.status == found_empty);
  static_assert(v1.ptr == nullptr);
  static_assert(v1.size == 0);
  constexpr auto v2 = non_local_lookup_core<unsigned char>(u8"resources/a/b/empty");
  static_assert(v2.status == found_empty);
  static_assert(v2.ptr == nullptr);
  static_assert(v2.size == 0);

  constexpr auto v3 = non_local_lookup_core<char>("resources/a/b/empty");
  static_assert(v3.status == found_empty);
  static_assert(v3.ptr == nullptr);
  static_assert(v3.size == 0);
  constexpr auto v4 = non_local_lookup_core<char>(L"resources/a/b/empty");
  static_assert(v4.status == found_empty);
  static_assert(v4.ptr == nullptr);
  static_assert(v4.size == 0);
  constexpr auto v5 = non_local_lookup_core<char>(u8"resources/a/b/empty");
  static_assert(v5.status == found_empty);
  static_assert(v5.ptr == nullptr);
  static_assert(v5.size == 0);

  constexpr auto v6 = non_local_lookup_core<std::byte>("resources/a/b/empty");
  static_assert(v6.status == found_empty);
  static_assert(v6.ptr == nullptr);
  static_assert(v6.size == 0);
  constexpr auto v7 = non_local_lookup_core<std::byte>(L"resources/a/b/empty");
  static_assert(v7.status == found_empty);
  static_assert(v7.ptr == nullptr);
  static_assert(v7.size == 0);
  constexpr auto v8 = non_local_lookup_core<std::byte>(u8"resources/a/b/empty");
  static_assert(v8.status == found_empty);
  static_assert(v8.ptr == nullptr);
  static_assert(v8.size == 0);

  return true;
}

#undef STR_PREFIX_
#undef STR_PREFIX

static_assert(does_not_exist());
static_assert(not_depended_on());
static_assert(empty());
