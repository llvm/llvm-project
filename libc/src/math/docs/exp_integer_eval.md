# Integer-assisted binary64 exponential

The RV32 selector requires integer multiplication support (M or Zmmul), no
hardware double-precision arithmetic (D or Zdinx),
and only with nearest-only, no-exception and no-errno math configurations. The
implementation retains `double_eval::exp` as a fallback. Other configurations
retain their existing public selection. The internal adapter can also be tested
with all four rounding modes on a host with a floating-point environment.
An explicit `LIBC_MATH_EXP_IMPL` override is honored: `double_eval` retains the
existing implementation and `integer_eval` selects the hybrid for testing.
Unsupported implementation tokens remain compile-time errors.

The integer kernel accepts finite inputs with `2^-53 <= |x| < 700`, for which the
outputs are normal. The hybrid delegates the boundary `|x|=2^-53` directly to
the fallback, avoiding a known nearest-mode ambiguity. Tiny inputs, extreme
finite inputs, infinities and NaNs are
handled by the existing implementation. An ambiguous interval also takes that
fallback. No unresolved status escapes the public function.

## Range reduction

The binary64 significand and scale represent the input exactly. Choose an integer
`k` by truncating an integer approximation to `|x| * log2(e) * 128`, and restore
its sign. The approximation need not choose a particular nearest integer: the
actual residual is enclosed separately. Reject `|k| > 131072`.

Write `k = 128*q + i`, with `0 <= i < 128`, and use

    exp(x) = 2^q * 2^(i/128) * exp(r), r = x - k*ln(2)/128.

`LN2_LO / 2^96 < ln(2) < (LN2_LO+1)/2^96`. Compute the residual endpoints on a
Q96 grid, including all shift rounding, and narrow them outward to Q64. Reject
unless both magnitudes are at most `floor(2^64/184)`. Thus `|r| <= 1/184`.

The `LOG2_E_63` constant is the floor of `2^63/ln(2)`. Its floor is established
using independent Q192 rational bounds on ln(2). It chooses a reduction integer;
it does not supply the residual error bound.

## Constants

`libc/utils/mathtools/exp_integer_constants.py` reproduces/checks the constants
without a floating-point dependency.

For each table row i, the lower integer T satisfies

    T^128 <= 2^(128*64+i) < (T+1)^128.

The table value is consequently enclosed in `[T,T+1]/2^64`, with i=0 exact.
The ln(2) bounds use `2*atanh(1/3)` and a rational geometric bound on the positive
omitted tail. Reciprocal-factorial coefficients are the exact integer floor of
`2^64/n!`; coefficients n=0,1,2 are exact and others use floor/ceiling endpoints.

## Polynomial and rounding errors

Evaluate the degree-six Taylor polynomial by outward-rounded interval Horner
arithmetic. Signed residual endpoints are handled explicitly. The Horner
intermediates are positive on this small interval. Before subtracting a negative
product, the implementation verifies that the unsigned subtraction is valid.

The absolute omitted tail is bounded by

    |r|^7 / (7! * (1 - |r|/8)).

At `|r|=1/184` this is less than `2^-64`, as checked with exact rationals by the
constant generator. Widen the evaluated polynomial by one Q64 unit in each
direction. All coefficient and arithmetic rounding errors are already included
in the interval calculation.

The interval contains `exp(r)`. Multiplication by the positive table enclosure
then gives an enclosure of `exp(x) / 2^q`.

## Integer capacity

Use native `__uint128_t` on GCC targets that provide it, through the existing
`UInt128` alias. Retain `UInt<128>` for other compilers and targets. This avoids
GCC generic full-product overhead. Shifts, including `one_at`, explicitly guard
counts outside [0,127] on every backend. All backends use integer comparisons for
the fixed capacity checks. Clang's limb backend is retained, but its generated
host code is intentionally changed by the hot-path optimization below.
The significand/log2(e) product uses at most 117 bits; products of the reduction
integer and Q96 ln(2) bounds use at most 114 bits. Input shifts use at most 106
bits. The Horner loop explicitly rejects intermediates longer than 65 bits, so
its products with residual magnitudes shorter than 57 bits fit in 122 bits.

The six Horner steps are compile-time expanded with residual signs selected
once, retaining the existing interval operations and capacity checks. Steps
n=5 through n=2 store their positive endpoints in one 64-bit limb; steps n=1
and n=0 retain the 65-bit representation. To check this narrowing with exact
integers, start U6=floor(2^64/6!)+1 and recursively bound each step by

    Pn=ceil(U(n+1)*floor(2^64/184)/2^64),
    Ln=floor(2^64/n!)-Pn,
    Un=ceil(2^64/n!)+Pn.

Every Ln is positive. U2=9240103771322441165 is below 2^64, as are U3 through
U6; U1 and U0 remain below 2^65. These are worst-case bounds, not assumptions
based on the sampled inputs.
The constant generator checks these bounds and the reduction shift counts as
part of its certificate, including when invoked with `--check`.

For a Q64 product with operand widths (65,57) or (57,64), compute

    floor(a*b/2^64) = high64(low64(a)*b) + high64(a)*b.

The low 64 bits of the same product supply the discarded-bit test for upward
rounding. The rounded result fits 64 bits in both supported operand ranges.
This avoids general full-width cross-products without discarding numerical
information. The accepted input-exponent range also simplifies the reduction
integer to a high-limb right shift; the two ln(2) products share the exact
product k*LN2_LO, using k*(LN2_LO+1)=k*LN2_LO+k.

The final scale-product would otherwise need up to 130 bits. Split it into
`exp(r)=1+u` and `table=1+v`, so the scaled result is the sum of the table, u and
u*v. Reject `|u| >= 2^-7`; v is in [0,1). Therefore the integer correction product
fits in 121 bits. Apply directed rounding with the appropriate sign. No
multiplication relies on unsigned truncation for its numerical result.

Round both final endpoints on the destination binary64 grid using the requested
mode. Accept only identical encodings; otherwise use the existing fallback.
The normal-result rounding helper also rejects values outside its normal range.
It recognizes scaled endpoints in [1/2,4) without a bit scan and rounds the
discarded low-limb field directly. Its general integer-rounding path remains
available for other input shapes.

When both endpoints have the same high limb in [0,3] and lie in [1/2,4),
their bit widths coincide. Round the pair with a shared width, exponent,
discarded-bit mask and high-limb contribution. The shift count is then 11,
12 or 13, and each unrounded significand lies in [2^52,2^53). Apply the same
ties-to-even or directed increment independently to both endpoints, and still
require equal rounded significands. Packing by exponent addition propagates a
significand carry exactly; reject an encoding at or above positive infinity.
Endpoint pairs outside this common shape retain the two general conversions.
This changes neither endpoint rounding nor candidate acceptance. No Horner
capacity or subtraction checks are removed.

## Environment and validation scope

For an accepted nonzero ordinary input, the internal host adapter raises
FE_INEXACT through the existing conditional helper and preserves errno. It
cannot produce an exact result: a nonzero binary64 input is algebraic, whereas
its exponential is transcendental and hence not representable in binary64.
The adapter
checks the special/tiny input range before using the rounding-mode helper, so
zero and infinity do not acquire an inexact flag from that helper. All exceptional
behavior is delegated to the existing implementation. The selected RV32
configuration compiles out exception/errno handling and assumes nearest mode.

This design argument does not constitute a machine-checked proof of the C++
implementation. The implementation needs MPFR differential tests, environment
checks, and target validation. Instruction counts in an emulator are not measured
hardware cycles and do not establish a speedup on every embedded processor.
