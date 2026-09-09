```{title} clang-tidy - performance-inefficient-algorithm
```

# performance-inefficient-algorithm

Warns on inefficient use of STL algorithms on associative containers.

Associative containers implement some of the algorithms as methods which
should be preferred to the algorithms in the algorithm header. The methods
can take advantage of the order of the elements.

```cpp
std::set<int> s;
auto it = std::find(s.begin(), s.end(), 43);

// becomes

auto it = s.find(43);
```

```cpp
std::set<int> s;
auto c = std::count(s.begin(), s.end(), 43);

// becomes

auto c = s.count(43);
```

In a `std::set` or `std::multiset` that `std::less` orders, finding the first
element past a fixed bound can use binary search instead of a linear scan:

```cpp
std::set<int> s;
auto it = std::find_if(s.begin(), s.end(), [](int v) { return v > 43; });

// becomes

auto it = s.upper_bound(43);
```

`>` maps to `upper_bound` and `>=` to `lower_bound`. Only built-in comparison
operators are rewritten.
