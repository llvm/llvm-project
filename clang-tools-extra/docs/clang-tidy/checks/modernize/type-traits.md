```{title} clang-tidy - modernize-type-traits
```

# modernize-type-traits

Converts standard library type traits of the form `traits<...>::type` and
`traits<...>::value` into `traits_t<...>` and
`traits_v<...>` respectively.

Also suggests converting `std::remove_cv_t<std::remove_reference_t<...>` into
`std::remove_cvref_t<...>` when targeting C++20 or above.

For example:

```c++
std::is_integral<T>::value
std::is_same<int, float>::value
typename std::add_const<T>::type
std::make_signed<unsigned>::type

std::remove_cv_t<std::remove_reference_t<int>>
```

Would be converted into:

```c++
std::is_integral_v<T>
std::is_same_v<int, float>
std::add_const_t<T>
std::make_signed_t<unsigned>

std::remove_cvref_t<int>
```

## Options

(modernize-type-traits-ignore-macros)=

````{option} IgnoreMacros
If `true` don't diagnose traits defined in macros.

Note: Fixes will never be emitted for code inside of macros.

```c++
#define IS_SIGNED(T) std::is_signed<T>::value
```

Default is `false`.
````

## Limitations

Does not currently diagnose uses of type traits with nested name
specifiers (e.g. `std::chrono::is_clock`,
`std::chrono::treat_as_floating_point`).
