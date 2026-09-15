#!/usr/bin/env python3
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

"""Generates the reference data used by the [sf.cmath] tests.

L^m_n(x) is a polynomial in x with rational coefficients, and every binary
floating-point number is a rational, so the exact mathematical value of
assoc_laguerre(n, m, x) is a Fraction. No external library, no extended-precision
library and no approximation is involved: the value below is the exact one, rounded
once to the requested number of decimal digits.

That value is computed with the same three-term recurrence Boost.Math evaluates, so a
mistake in it would agree with the implementation under test and go unnoticed. Every
value is therefore also computed from the closed form, which shares no arithmetic with
the recurrence, and the two are required to agree exactly.

That only works for the polynomial members of [sf.cmath] (assoc_laguerre, laguerre,
assoc_legendre, legendre, hermite). The transcendental ones (beta, the elliptic
integrals, expint, the Bessel family, riemann_zeta) have no such closed form and
will need a high-precision reference instead.

Sampling. The roots of L^m_n all lie in (0, ~3.3n + 2m), so the interesting range of
x scales with n and m rather than being fixed. Sampling x at fixed absolute offsets
would land every point in the smooth tail for small n and miss the tail entirely for
large n. Each (n, m) therefore gets its own points, in three families:

  * linear in x/largest_root over the oscillatory band, where the roots and the
    cancellation live,
  * logarithmic from there up to just below the overflow threshold of `float`, since
    the tail spans tens of decades,
  * pseudo-random with a full 24-bit significand. The two grids above only ever
    produce x with a handful of significant bits, which never exercises rounding of
    the argument itself.

Every x is snapped to the nearest `float`, so a single decimal literal is exact in
every format the overloads use, and x == 0 is always included.

The output is committed, and CI regenerates it and rejects the patch if the result
differs, so this script must be deterministic. random.Random is seeded and only
getrandbits() is used; iteration order never depends on a set or a dict.

Usage
-----

The supported way to run this is the build target, from the repository root:

    ninja -C <build> libcxx-generate-files

That regenerates every generated file in the tree, which is what CI checks, so it is
also the way to confirm a change here leaves nothing stale. To run this generator by
itself:

    libcxx/utils/generate_sf_cmath_test_data.py \
        libcxx/test/std/numerics/c.math/sf.cmath/assoc_laguerre_data.h

Given no argument it writes to stdout, which is how to preview a change to the
sampling before overwriting the committed header:

    libcxx/utils/generate_sf_cmath_test_data.py \
        | diff -u libcxx/test/std/numerics/c.math/sf.cmath/assoc_laguerre_data.h -

Only the Python standard library is used; nothing needs installing. What gets
tabulated is set by the MAX_DEGREE, ORDERS, MAX_CONDITION and SIGNIFICANT_DIGITS
constants below. Changing any of them changes the header, which then has to be
regenerated and committed in the same patch.
"""

import math
import os
import random
import struct
import sys
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction

# Highest degree to tabulate. The degree is what drives both the cost and the
# conditioning, and n > 20 adds little beyond what n <= 20 already covers.
MAX_DEGREE = 20

# Orders to tabulate. m == 0 is a separate code path in the implementation, so it is
# present for every degree; beyond that the recurrence varies smoothly with m and a
# sparse set is enough.
ORDERS = (0, 1, 2, 3, 5, 7, 10)

# Upper bound on max|intermediate| / |result| for a row to be kept. Above it the
# result is so close to a root that its computed value carries no information.
MAX_CONDITION = 64

# Enough to round-trip through IEEE binary128, the widest `long double` in use.
SIGNIFICANT_DIGITS = 36

SEED = 0x5F3759DF

FLT_MAX = 2.0**128 - 2.0**104


def assoc_laguerre_exact(n, m, x):
    """Returns the exact L^m_n(x) and max|intermediate| / |result| for rational x.

    Evaluated with the same three-term recurrence the implementation uses, so the
    ratio reported here describes that algorithm's cancellation and not merely the
    conditioning of the polynomial.
    """
    if n == 0:
        return Fraction(1), Fraction(1)

    previous, current = Fraction(1), Fraction(m + 1) - x
    largest = max(abs(previous), abs(current))
    for k in range(1, n):
        previous, current = (
            current,
            ((2 * k + m + 1 - x) * current - (k + m) * previous) / (k + 1),
        )
        largest = max(largest, abs(current))

    if current == 0:
        return current, None
    return current, largest / abs(current)


def assoc_laguerre_explicit(n, m, x):
    """Returns the exact L^m_n(x) from the closed form, independently of the recurrence.

    L^m_n(x) = sum over k of (-1)^k * C(n + m, n - k) * x^k / k!
    """
    total = Fraction(0)
    power = Fraction(1)
    factorial = 1
    for k in range(n + 1):
        total += Fraction((-1) ** k * math.comb(n + m, n - k), factorial) * power
        power *= x
        factorial *= k + 1
    return total


def nearest_float(value):
    """Rounds a Fraction to the nearest `float`, exactly representable everywhere."""
    return Fraction(struct.unpack("f", struct.pack("f", float(value)))[0])


def largest_root_bound(n, m):
    """Approximates the largest root of L^m_n, which bounds the oscillatory band."""
    return 3.3 * n + 2.0 * m + 2.0


def float_overflow_bound(n):
    """Approximates the x at which |L^m_n(x)| leaves the range of `float`.

    Past its largest root L^m_n(x) is dominated by x^n / n!, which gives the bound
    directly. It only has to be about right: the caller keeps a wide margin, and a
    row whose value does not fit is dropped after the fact anyway.
    """
    if n == 0:
        return math.inf
    return math.exp((math.log(FLT_MAX) + math.lgamma(n + 1)) / n)


def sample_points(n, m, rng):
    """Returns the x values to tabulate for one (n, m), in ascending order."""
    root = largest_root_bound(n, m)
    # Halving the overflow bound keeps the tail clear of it; capping at a multiple of
    # the root span keeps the low degrees, whose bound is astronomically large, from
    # spending all their points on the featureless tail.
    limit = min(float_overflow_bound(n) * 0.5, root * 24.0)

    # L^m_n is constant in x for n == 0. A handful of points documents that; a full
    # grid would be 84 rows all asserting the same 1.
    if n == 0:
        points = {Fraction(0), nearest_float(root), nearest_float(limit)}
        return sorted(points)

    points = {Fraction(0)}
    points |= {nearest_float(root * i / 6.0) for i in range(1, 6)}
    if limit > root:
        points |= {
            nearest_float(root * (limit / root) ** (i / 4.0)) for i in range(1, 5)
        }
    points |= {nearest_float(limit * rng.getrandbits(24) / 2.0**24) for _ in range(3)}
    return sorted(points)


def format_exact(value):
    """Formats a Fraction with a power-of-two denominator, without rounding."""
    with localcontext() as context:
        context.prec = 80
        decimal = (Decimal(value.numerator) / Decimal(value.denominator)).normalize()
    assert Fraction(decimal) == value, f"{value} is not exactly representable"
    return format_literal(decimal)


def format_rounded(value):
    """Formats a Fraction, correctly rounded to SIGNIFICANT_DIGITS digits."""
    with localcontext() as context:
        context.prec = SIGNIFICANT_DIGITS
        context.rounding = ROUND_HALF_EVEN
        decimal = (Decimal(value.numerator) / Decimal(value.denominator)).normalize()
    return format_literal(decimal)


def format_literal(decimal):
    """Renders a Decimal as a `long double` literal in a single canonical form."""
    sign, digits, exponent = decimal.as_tuple()
    text = "".join(map(str, digits))
    leading = exponent + len(text) - 1
    mantissa = text if len(text) == 1 else f"{text[0]}.{text[1:]}"
    return f"{'-' if sign else ''}{mantissa}e{leading:+03d}L"


def generate_rows():
    """Yields one formatted table row per kept (n, m, x)."""
    rng = random.Random(SEED)
    for n in range(MAX_DEGREE + 1):
        for m in ORDERS:
            for x in sample_points(n, m, rng):
                value, condition = assoc_laguerre_exact(n, m, x)
                assert value == assoc_laguerre_explicit(n, m, x), f"L^{m}_{n}({x})"

                # A zero has no meaningful relative error, a value outside the range
                # of `float` would overflow the narrowest overload (the dedicated
                # tests cover that), and an ill-conditioned point says nothing about
                # the implementation.
                if value == 0 or condition is None:
                    continue
                if abs(value) >= FLT_MAX or condition > MAX_CONDITION:
                    continue

                yield f"    {{{n}, {m}, {format_exact(x)}, {format_rounded(value)}}},"


HEADER_TEMPLATE = """// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// WARNING, this entire header is generated by
// utils/generate_sf_cmath_test_data.py
// DO NOT MODIFY!

#ifndef TEST_SF_CMATH_ASSOC_LAGUERRE_DATA_H
#define TEST_SF_CMATH_ASSOC_LAGUERRE_DATA_H

// Reference values for std::assoc_laguerre, covering degrees 0 to {max_degree} and orders
// {orders}.
//
// L^m_n(x) is a polynomial in x with rational coefficients and every binary
// floating-point number is a rational, so `expected` is the exact mathematical value
// rounded once to {digits} decimal digits -- enough to round-trip through IEEE binary128.
// Each value is computed twice, by the three-term recurrence and by the closed form, and
// the generator requires the two to agree exactly.
// `x` is a `float` written out exactly, so both literals are exact in every format the
// overloads use.
//
// Points too close to a root to be informative are left out: a row is kept only if the
// recurrence's max|intermediate| / |result| is at most {max_condition}. Values outside the range
// of `float` are left out as well, so every row applies to every overload; the range
// errors they would raise are covered by the dedicated tests instead.
//
// Do not compare against `expected` with ==. The implementation evaluates a three-term
// recurrence, which loses accuracy to cancellation wherever no wider type is available
// to evaluate in, and the literals below are `long double` and round twice on their way
// to `float`. See the tolerance the test applies.

struct AssocLaguerreTestCase {{
  unsigned n;
  unsigned m;
  long double x;
  long double expected;
}};

inline constexpr AssocLaguerreTestCase assoc_laguerre_test_data[] = {{
{rows}
}};

#endif // TEST_SF_CMATH_ASSOC_LAGUERRE_DATA_H
"""


def generate_all():
    """Returns the complete header."""
    return HEADER_TEMPLATE.format(
        max_degree=MAX_DEGREE,
        orders=", ".join(str(m) for m in ORDERS),
        digits=SIGNIFICANT_DIGITS,
        max_condition=MAX_CONDITION,
        rows="\n".join(generate_rows()),
    )


USAGE = """\
Usage:
{name} [output-header]

Writes the [sf.cmath] reference data header, or to stdout if no path is given.
Normally run through the `libcxx-generate-files` build target rather than directly.
"""


def main(argv):
    """Writes the header to the path in argv, or to stdout when argv is empty."""
    if len(argv) > 1:
        sys.stderr.write(USAGE.format(name=os.path.basename(__file__)))
        return 1

    header = generate_all()
    if not argv:
        sys.stdout.write(header)
        return 0

    # The header has to be ASCII: libcxx/utils/ci/run-buildbot rejects a patch that puts
    # anything else under libcxx/test. Writing it as ASCII makes that a failure here, where
    # the cause is visible, rather than one in CI.
    with open(argv[0], "w", encoding="ascii") as output:
        output.write(header)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
