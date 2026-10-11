```{title} clang-tidy - bugprone-unchecked-expected-access
```

# bugprone-unchecked-expected-access

*Note*: This check uses a flow-sensitive static analysis to produce its
results. Therefore, it may be more resource intensive (RAM, CPU) than the
average clang-tidy check.

This check identifies unsafe accesses to values and errors contained in
`std::expected<T, E>` objects.

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
```

## Limitations

The check does not yet model some member functions of `std::expected`. These
are known, temporary limitations:

- `swap`. For example, the check does not know that after `a.swap(b)`, `a`
  holds a value if `b` held one before.
- The monadic operations `and_then`, `or_else`, `transform` and
  `transform_error`. The check does not relate the state of their result to the
  state of the object they are called on, so accessing the result without a
  check of its own is diagnosed, even when it is safe:

```cpp
int f(std::expected<int, Error> e) {
  if (!e)
    return 0;
  auto r = e.transform([](int v) { return v + 1; });
  return *r; // false positive: `r` holds a value because `e` does.
}

void g(std::expected<int, Error> e) {
  if (!e)
    report(e.transform_error(describe).error()); // false positive: the result
                                                 // holds an error because `e`
                                                 // does.
}
```
