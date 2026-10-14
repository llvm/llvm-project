```{title} clang-tidy - modernize-use-std-interpolation
```

# modernize-use-std-interpolation

Replaces manual midpoint and linear interpolation calculations.

The check suggests `std::midpoint` from `<numeric>` and `std::lerp` from `<cmath>`
to express the calculation's intent and avoid intermediate overflow. It requires
C++20 or later.

For example:

```c++
int midpoint(int a, int b) {
  return (a + b) / 2;
}

double interpolate(double a, double b, double t) {
  return a + (b - a) * t;
}
```

becomes:

```c++
#include <cmath>
#include <numeric>

int midpoint(int a, int b) {
  return std::midpoint(a, b);
}

double interpolate(double a, double b, double t) {
  return std::lerp(a, b, t);
}
```

The following expressions are recognized:

| Expression | Replacement |
| --- | --- |
| `(a + b) / 2` | `std::midpoint(a, b)` |
| `a + (b - a) / 2` | `std::midpoint(a, b)` |
| `(a + b) * 0.5` | `std::midpoint(a, b)` |
| `a + (b - a) * 0.5` | `std::midpoint(a, b)` |
| `a + (b - a) * t` | `std::lerp(a, b, t)` |
| `(1 - t) * a + t * b` | `std::lerp(a, b, t)` |

Sum midpoint formulas must have two endpoints. Ungrouped addition or subtraction
chains such as `(a + b + 1) / 2` are excluded. Parenthesized endpoints such as
`((a + b) + c) / 2` are supported.

## Changes in numerical behavior

Replacing integer `(a + b) / 2` can change rounding. Integer division truncates
toward zero, whereas `std::midpoint(a, b)` rounds toward `a`. For example,
`(2 + 1) / 2` yields `1`, but `std::midpoint(2, 1)` yields `2`.
Unsigned difference formulas can also change results when the subtraction wraps.

Floating-point replacements can change rounding, overflow handling, signed zero,
and results involving infinities or NaNs. These replacements adopt the numerical
behavior of the standard library facilities; they do not promise identical
results to the original arithmetic.

## Options

### IncludeStyle

A string specifying which include ordering convention to use: `llvm` or
`google`. The default is `llvm`.
