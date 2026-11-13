```{title} clang-tidy - bugprone-std-namespace-modification
```

# bugprone-std-namespace-modification

Warns on modifications of the `std` or `posix` namespaces which can
result in undefined behavior.

The `std` (or `posix`) namespace is allowed to be extended with (class or
function) template specializations that depend on an user-defined type (a type
that is not defined in the standard system headers).

The check detects the following (user provided) declarations in namespace
`std` or `posix`:

- Anything that is not a template specialization.
- Explicit specializations of any standard library function template or class
  template, if it does not have any user-defined type as template argument.
- Explicit specializations of any member function of a standard library class
  template.
- Explicit specializations of any member function template of a standard
  library class or class template.
- Explicit or partial specialization of any member class template of a standard
  library class or class template.

A partial specialization of a standard library class template whose template
arguments do not name a user-defined type is not diagnosed if its constraints
restrict it to user-defined types. The check evaluates the constraints for a
set of probe arguments that are not user-defined:

- the types `int`, `unsigned`, `long`, `double`, `bool`, `char`, `void`,
  `std::nullptr_t`, `int *`, `int &`, `int &&`, `const int`, `int[1]` and
  `void()`,
- the types `std::byte`, `std::true_type`, `std::string`, `std::vector<int>`,
  `std::tuple<int>`, `std::pair<int, int>`, `std::optional<int>`,
  `std::map<int, int>` and `std::array<int, 1>`, if they are declared in the
  translation unit,
- an empty pack or a single probe type for a parameter pack, and
- `0` and `1` for a non-type template parameter of integral or enumeration
  type.

If no combination of probe arguments satisfies the constraints, the
specialization is considered to depend on a user-defined type and is not
diagnosed. Constraints that cannot be evaluated, and specializations with other
kinds of template parameters, are diagnosed as before.

This is a heuristic. A constraint that is only satisfied by a type outside the
probe set, such as `std::is_same_v<T, std::errc>`, is not diagnosed. Since
the standard library probe types depend on the included headers, the result for
a specialization in a header file can differ between translation units and
between standard library implementations.

Examples:

```c++
namespace std {
  int x; // warning: modification of 'std' namespace can result in undefined behavior [bugprone-dont-modify-std-namespace]
}

namespace posix::a { // warning: modification of 'posix' namespace can result in undefined behavior
}

template <>
struct ::std::hash<long> { // warning: modification of 'std' namespace can result in undefined behavior
  unsigned long operator()(const long &K) const {
    return K;
  }
};

struct MyData { long data; };

template <>
struct ::std::hash<MyData> { // no warning: specialization with user-defined type
  unsigned long operator()(const MyData &K) const {
    return K.data;
  }
};

template <class T>
concept MyDataLike = std::derived_from<T, MyData>;

template <MyDataLike T>
struct std::hash<T> { // no warning: constrained to user-defined types
  unsigned long operator()(const T &K) const {
    return K.data;
  }
};

template <std::integral T>
struct std::hash<T *> { // warning: modification of 'std' namespace can result in undefined behavior
  unsigned long operator()(T *) const {
    return 0;
  }
};

namespace std {
  template <>
  void swap<bool>(bool &a, bool &b); // warning: modification of 'std' namespace can result in undefined behavior

  template <>
  bool less<void>::operator()<MyData &&, MyData &&>(MyData &&, MyData &&) const { // warning: modification of 'std' namespace can result in undefined behavior
    return true;
  }
}
```

## References

This check corresponds to the CERT C++ Coding Standard rule
[DCL58-CPP. Do not modify the standard namespaces](https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/declarations-and-initialization-dcl/dcl58-cpp/).
