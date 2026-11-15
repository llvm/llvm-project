```{title} clang-tidy - bugprone-unchecked-expected-access
```

# bugprone-unchecked-expected-access

*Note*: This check uses a flow-sensitive static analysis to produce its
results. Therefore, it may be more resource intensive (RAM, CPU) than the
average clang-tidy check.

This check identifies unsafe accesses to values and errors contained in
`std::expected<T, E>` objects.

The check also supports `tl::expected<T, E>` from
[TartanLlama/expected](https://github.com/TartanLlama/expected), and
everything below applies to it as well. Its `map` and `map_error` member
functions are treated like `transform` and `transform_error`.

An access to the value of a `std::expected<T, E>` occurs when one of its
`value`, `operator*`, or `operator->` member functions is invoked. The check
considers these member functions as equivalent, even though `value` throws
`std::bad_expected_access` while `operator*` and `operator->` have undefined
behavior when the object holds an error.

An access to the value of a `std::expected<T, E>` is considered safe if and
only if code in the local scope (for example, a function body) ensures that
the object holds a value in all possible execution paths that can reach the
access. That should happen through an explicit check, using the `has_value`
member function or the conversion to `bool`.

Likewise, an access to the error of a `std::expected<T, E>` occurs when its
`error` member function is invoked, which has undefined behavior when the
object holds a value. Such an access is considered safe if and only if code in
the local scope ensures that the object does *not* hold a value in all
possible execution paths that can reach the access.

## Unsafe access patterns

```cpp
void f(std::expected<int, Error> e) {
  use(*e); // unsafe: it is unclear whether `e` holds a value.
}

void g(std::expected<int, Error> e) {
  if (!e) {
    use(e.value()); // unsafe: `e` holds an error here.
  }
}

void h(std::expected<int, Error> e) {
  report(e.error()); // unsafe: it is unclear whether `e` holds an error.
}
```

## Safe access patterns

```cpp
void f(std::expected<int, Error> e) {
  if (e.has_value()) {
    use(*e);
  }
}

void g(std::expected<int, Error> e) {
  if (!e)
    return;
  use(e.value());
}

int h(std::expected<int, Error> e) {
  return e ? *e : 0;
}

void i(std::expected<int, Error> e) {
  if (!e)
    report(e.error());
}

int j(std::expected<int, Error> e) {
  if (!e)
    e.emplace(0);
  return *e;
}

int k(std::expected<int, Error> a, std::expected<int, Error> b) {
  if (a == b && b)
    return *a; // equal objects either both hold a value or both an error.
  return 0;
}

int l(std::expected<int, Error> e) {
  if (e == 42)
    return *e; // only an expected holding a value compares equal to a value.
  return 0;
}

void m(std::expected<int, Error> e) {
  if (e == std::unexpected(Error::NotFound))
    report(e.error()); // only an expected holding an error compares equal
                       // to an unexpected.
}

int n() {
  std::expected<int, Error> e; // a default-constructed expected holds a value.
  return *e;
}

void o(std::expected<int, Error> e) {
  e = std::unexpected(Error::NotFound);
  report(e.error());
}

int p(std::expected<int, Error> e) {
  if (!e)
    return 0;
  // `transform` and `transform_error` hold a value if and only if `e` does.
  return *e.transform([](int v) { return v + 1; });
}
```

## Options

```{option} IgnoreValueCalls
If set to `true`, the check does not diagnose calls to `expected::value()`,
which throws `bad_expected_access` when there is no value. Diagnostics for
`operator*()`, `operator->()` and `error()` remain enabled. Default is `false`.
```

## Limitations

1. The check does not yet model some member functions of `std::expected`. These
are known, temporary limitations:

- `swap`. For example, the check does not know that after `a.swap(b)`, `a`
  holds a value if `b` held one before.
- The monadic operations `and_then` and `or_else`. The check does not use the
  fact that `and_then` keeps an error and `or_else` keeps a value, so accessing
  their result without a check of its own is diagnosed, even when it is safe:

```cpp
void f(std::expected<int, Error> e) {
  if (!e)
    report(e.and_then(validate).error()); // false positive: the result holds
                                          // an error because `e` does.
}

int g(std::expected<int, Error> e) {
  if (!e)
    return 0;
  return *e.or_else(recover); // false positive: the result holds a value
                              // because `e` does.
}
```

2. **GoogleTest assertions** such as `ASSERT_TRUE(e.has_value())` are not recognized
yet; support for them is planned.

3. **Lambdas.** The check does not currently report unsafe expected accesses in lambdas; support for analyzing is planned.

4. **Derived classes.** Classes derived from `std::expected` or `tl::expected` are not analyzed by this check.
