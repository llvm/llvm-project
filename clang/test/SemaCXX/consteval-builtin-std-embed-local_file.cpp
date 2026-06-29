// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-c++2d-extensions -Wno-c++23-extensions -Wnoc23-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wno-c++2d-extensions -Wnoc23-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wno-c++2d-extensions -Wnoc23-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++2d -fsyntax-only -Wnoc23-extensions --embed-dir=%S/Inputs %s
// expected-no-diagnostics

#depend __FILE__

namespace std {
  enum class byte : unsigned char {};
}

template <typename I0, typename S0, typename I1, typename S1>
constexpr bool byte_equal (I0 first0, S0 last0, I1 first1, S1 last1) {
  for (;first0 != last0 || first1 != last1; ++first0, ++first1) {
    if ((unsigned char)*first0 != (unsigned char)*first1) {
      return false;
    }
  }
  return first0 == last0 && first1 == last1;
}

inline constexpr decltype(sizeof(0)) sentinel_size = 44203;
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
  found_empty,
};

template <typename T, typename C, decltype(sizeof(0)) N>
consteval inspect<T> local_lookup_core (const C (&name)[N]) {
  inspect<T> result;
  result.ptr = __builtin_std_embed(0b000, result.status, result.size, result.ptr, (N)-1, name, 0);
  return result;
}

consteval bool local_file () {
  static constexpr const unsigned char file[] = {
#embed __FILE__
  };
  static constexpr const decltype(sizeof(0)) file_size = sizeof(file);
  constexpr auto v0 = local_lookup_core<unsigned char>(__FILE__);
  static_assert(v0.status == found);
  static_assert(byte_equal(v0.ptr, v0.ptr + v0.size, &file[0], &file[0] + file_size));
  static_assert(v0.size == file_size);

  constexpr auto v1 = local_lookup_core<char>(__FILE__);
  static_assert(v1.status == found);
  static_assert(byte_equal(v1.ptr, v1.ptr + v1.size, &file[0], &file[0] + file_size));
  static_assert(v1.size == file_size);

  constexpr auto v2 = local_lookup_core<std::byte>(__FILE__);
  static_assert(v2.status == found);
  static_assert(byte_equal(v2.ptr, v2.ptr + v2.size, &file[0], &file[0] + file_size));
  static_assert(v2.size == file_size);

  return true;
}

static_assert(local_file());
