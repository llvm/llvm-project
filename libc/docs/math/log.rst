.. _log_algorithm:

========================
Log/Log10/Log2 Algorithm
========================

.. default-role:: math

In this short note, we will discuss in detail about the computation of
:math:`\log(x)` function, with double precision inputs, in particular, the range
reduction steps and error analysis.  The algorithm is broken down into 2 main
phases as follow:

1. Fast phase:

  a. Range reduction
  b. Polynomial approximation
  c. Ziv's test

2. Accurate phase (if Ziv's test failed):

  a. Further range reduction
  b. Polynomial approximation


Fast phase
==========

Range reduction
---------------

Let `x = 2^{e_x} (1 + m_x)` be a normalized double precision number, in which
`-1074 \leq e_x \leq 1022` and `0 \leq m_x < 1` such that
`2^{52} m_x \in \mathbb{Z}`.

Then from the properties of logarithm:

.. math::
  \log(x) &= \log\left( 2^{e_x} (1 + m_x) \right) \\
          &= \log\left( 2^{e_x} \right) + \log(1 + m_x) \\
          &= e_x \log(2) + \log(1 + m_x)

the computation of `\log(x)` can be reduced to:

1. compute the product of `e_x` and `\log(2)`,
2. compute `\log(1 + m_x)` for `0 \leq m_x < 1`,
3. add step 1 and 2.

To compute `\log(1 + m_x)` in step 2, we can reduce the range further by finding
`r > 0` such that:

.. math::
  | r(1 + m_x) - 1 | < C \quad \quad \text{(R1)}

for small `0 < C < 1`.  Then if we let `u = r(1 + m_x) - 1`, `|u| < C`:

.. math::
  \log(1 + m_x) &= \log \left( \frac{r (1 + m_x)}{r} \right) \\
                &= \log(r (1 + m_x) ) - \log(r) \\
                &= \log(1 + u) - \log(r)

and step 2 can be computed with:

a. extract `r` and `-\log(r)` from look-up tables,
b. compute the reduced argument `u = r(1 + m_x) - 1`,
c. compute `\log(1 + u)` by polynomial approximation or further range reduction,
d. add step a and step c results.


How to derive `r`
-----------------

For an efficient implementation, we would like to use the first `M` significant
bits of `m_x` to look up for `r`.  In particular, we would like to find a value
of `r` that works for all `m_x` satisfying:

.. math::
  k 2^{-M} \leq m_x < (k + 1) 2^{-M} \quad \text{for some} \quad
  k = 0..2^{M} - 1. \quad\quad \text{(M1)}

Let `r = 1 + s`, then `u` can be expressed in terms of `s` as:

.. math::
  u &= r(1 + m_x) - 1 \\
    &= (1 + s)(1 + m_x) - 1 \\
    &= s m_x + s + m_x  &\quad\quad \text{(U1)} \\
    &= s (1 + m_x) + m_x \\
    &= m_x (1 + s) + s.

From the condition `\text{(R1)}`, `s` is bounded by:

.. math::
  \frac{-C - m_x}{1 + m_x} < s < \frac{C - m_x}{1 + m_x} \quad\quad \text{(S1)}.

Since our reduction constant `s` must work for all `m_x` in the interval
`I = \{ v: k 2^{-M} \leq v < (k + 1) 2^{-M} \}`, `s` is bounded by:

.. math::
  \sup_{v \in I} \frac{-C - v}{1 + v} < s < \inf_{v \in I} \frac{C - v}{1 + v}

For a fixed constant `|c| < 1`, let `f(v) = \frac{c - v}{1 + v}`, then its
derivative is:

.. math::
  f'(v) = \frac{(-1)(1 + v) - (1)(c - v)}{(1 + v)^2} = \frac{-1 - c}{(1 + v)^2}.

Since `|c| < 1`, `f'(v) < 0` for all `v \neq -1`, so:

.. math::
  \sup_{v \in I} f(v) &= f \left( \inf\{ v: v \in I \} \right)
                       = f \left( k 2^{-M} \right) \\
  \inf_{v \in I} f(v) &= f \left( \sup\{ v: v \in I \} \right)
                       = f \left( (k + 1) 2^{-M} \right)

Hence we have the following bound on `s`:

.. math::
  \frac{-C - k 2^{-M}}{1 + k 2^{-M}} < s \leq
  \frac{C - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}}. \quad\quad \text{(S2)}

In order for `s` to exist, we need that:

.. math::
  \frac{C - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}} > 
  \frac{-C - k 2^{-M}}{1 + k 2^{-M}}

which is equivalent to:

.. math::
  \quad\quad 2C - 2^{-M} + (2k + 1) 2^{-M} C > 0
  \iff C > \frac{2^{-M - 1}}{1 + (2k + 1) 2^{-M - 1}} \quad\quad \text{(C1)}.

Consider the case `C = 2^{-N}`.  Since `0 \leq k \leq 2^M - 1,` the right hand
side of `\text{(C1)}` is bounded by:

.. math::
  2^{-M - 1} > \frac{2^{-M - 1}}{1 + (2k + 1) 2^{-M - 1}} \geq
  \frac{2^{-M - 1}}{1 + (2^{M + 1} - 1) 2^{-M - 1}} > 2^{-M - 2}. 

Hence, from `\text{(C1)}`, being an exact power of 2, `C = 2^{-N}` is bounded below
by:

.. math::
  C = 2^{-N} \geq 2^{-M - 1}.

To make the range reduction efficient, we will want to minimize `C` (maximize
`N`) while keeping the required precision of `s`(`r`) as low as possible.  And
for that, we will consider the following two cases: `N = M + 1` and `N = M`.

Case 1 - `N = M + 1`
~~~~~~~~~~~~~~~~~~~~

When `N = M + 1`, `\text{(S2)}` becomes:

.. math::
  \frac{-2^{-M - 1} - k 2^{-M}}{1 + k 2^{-M}} < s <
  \frac{2^{-M - 1} - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}}.
  \quad\quad \text{(S2')}

This is an interval of length:

.. math::
  l &= \frac{2^{-M - 1} - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}} -
       \frac{-2^{-M - 1} - k 2^{-M}}{1 + k 2^{-M}} \\
    &= \frac{(2k + 1)2^{-2M - 1}}{(1 + k 2^{-M})(1 + (k + 1)2^{-M})}
    \quad\quad \text{(L1)}

As a function of `k`, the length `l` has its derivative with respect to `k`:

.. math::
  \frac{dl}{dk} =
  \frac{2^{2M + 1} - 2k(k + 1) - 1}
       {2^{4M}(1 + k 2^{-M})^2 (1 + (k + 1) 2^{-M})^2}

which is always positive for `0 \leq k \leq 2^M - 1`.  So for all
`0 < k < 2^{-M}` (`k = 0` will be treated differently in edge cases), and for
`M > 2`, `l` is bounded below by:

.. math::
  l > 2^{-2M}.

It implies that we can always find `s` with `\operatorname{ulp}(s) = 2^{-2M}`.
And from `\text{(U1)}`, `u = s(1 + m_x) + m_x`, its `ulp` is:

.. math::
  \operatorname{ulp}(u) &= \operatorname{ulp}(s) \cdot \operatorname{ulp}(m_x) \\
                        &= 2^{-2M} \operatorname{ulp}(m_x).

Since:

.. math::
  |u| < C = 2^{-N} = 2^{-M - 1},

Its required precision is:

.. math::
  \operatorname{prec}(u) &= \log_2(2^{-M-1} / \operatorname{ulp}(u)) \\
                         &= \log_2(2^{M - 1} / \operatorname{ulp}(m_x)) \\
                         &= M - 1 - \log_2(\operatorname{ulp}(m_x)).

This means that in this case, we cannot restrict `u` to be exactly representable
in double precision for double precision input `x` with `M > 2`.  Nonetheless,
for a reasonable value of `M`, we can have `u` exactly representable in double
precision for single precision input `x` (`\operatorname{ulp}(m_x) = 2^{-23}`)
such that `|u| < 2^{-M - 1}` using a look-up table of size `2^M`.

A particular formula for `s` can be derived from `\text{(S2')}` by the midpoint
formula:

.. math::
  s &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot \operatorname{midpoint}
       \left(-\frac{-2^{-M - 1} - k2^{-M}}{1 + k 2^{-M}},
       \frac{2^{-M-1} - (k + 1)2^{-M}}{1 + (k + 1) 2^{-M}}\right) \right) \\
    &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot \frac{1}{2} \left(
       \frac{-2^{-M - 1} - k2^{-M}}{1 + k 2^{-M}} +
       \frac{2^{-M - 1} + (k + 1)2^{-M}}{1 + (k + 1) 2^{-M}}
    \right) \right) \\
    &= 2^{-2M} \operatorname{round}\left( \frac{
       - \left(k + \frac{1}{2} \right) \left(2^M - k - \frac{1}{2} \right) }
       {(1 + k 2^{-N})(1 + (k + 1) 2^{-N})} \right) \\
    &= - 2^{-2M} \operatorname{round}\left( \frac{
       \left(k + \frac{1}{2} \right) \left(2^M - k - \frac{1}{2} \right) }
       {(1 + k 2^{-N})(1 + (k + 1) 2^{-N})} \right)  \quad\quad \text{(S3)}

The corresponding range and formula for `r = 1 + s` are:

.. math::
  \frac{1 - 2^{-M - 1}}{1 + k 2^{-M}} < r \leq
  \frac{1 + 2^{-M - 1}}{1 + (k + 1) 2^{-M}}

.. math::
  r &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot
       \operatorname{midpoint}\left( \frac{1 - 2^{-M - 1}}{1 + k 2^{-M}},
          \frac{1 + 2^{-M - 1}}{1 + (k + 1) 2^{-M}}\right) \right) \\
    &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot \frac{1}{2} \left(
       \frac{1 + 2^{-M-1}}{1 + (k + 1) 2^{-M}} + \frac{1 - 2^{-M-1}}{1 + k 2^{-M}}
    \right) \right) \\
    &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot \frac{
       1 + \left(k + \frac{1}{2} \right) 2^{-M} - 2^{-2M-2} }{(1 + k 2^{-M})
       (1 + (k + 1) 2^{-M})} \right)

Case 1 - `N = M`
~~~~~~~~~~~~~~~~

When `N = M`, `\text{(S2)}` becomes:

.. math::
  \frac{-(k + 1)2^{-M}}{1 + k 2^{-M}} < s < \frac{-k 2^{-M}}{1 + (k + 1) 2^{-M}}
  \quad\quad \text{(S2")}

This is an interval of length:

.. math::
  l &= \frac{- k 2^{-M}}{1 + (k + 1) 2^{-M}} -
       \frac{- (k + 1) 2^{-M}}{1 + k 2^{-M}} \\
    &= \frac{2^{-M} (1 + (2k + 1) 2^{-M})}{(1 + k 2^{-M})(1 + (k + 1)2^{-M})}
    \quad\quad \text{(L1')}

As a function of `k`, its derivative with respect to `k`:

.. math::
  \frac{dl}{dk} =
  -\frac{2^{-2M}(k(k + 1)2^{-M + 1} + 2^{-M} + 2k + 1)}
        {(1 + k 2^{-M})^2 (1 + (k + 1) 2^{-M})^2}

which is always negative for `0 \leq k \leq 2^M - 1`.  So for `M > 1`, `l` is
bounded below by:

.. math::
  l > \frac{2^{-M - 1} (3 - 2^{-M})}{2 - 2^{-M}} > 2^{-M - 1}.

It implies that we can always find `s` with `\operatorname{ulp}(s) = 2^{-M-1}`.
And from `\text{(U1)}`, `u = s(1 + m_x) + m_x`, its `ulp` is:

.. math::
  \operatorname{ulp}(u) &= \operatorname{ulp}(s) \cdot \operatorname{ulp}(m_x) \\
                        &= 2^{-M - 1} \operatorname{ulp}(m_x).

Since:

.. math::
  |u| < C = 2^{-N} = 2^{-M},

Its required precision is:

.. math::
  \operatorname{prec}(u) &= \log_2(2^{-M} / \operatorname{ulp}(u)) \\
                         &= \log_2(2 / \operatorname{ulp}(m_x)) \\
                         &= 1 - \log_2(\operatorname{ulp}(m_x)).

Hence, for double precision `x`, `\operatorname{ulp}(m_x) = 2^{-52}`, and the
precision needed for `u` is `\operatorname{prec}(u) = 53`, i.e., `u` can be
exactly representable in double precision.  And in this case, `s` can be
derived from `\text{(S2")}` by the midpoint formula:

.. math::
  s &= 2^{-M - 1} \operatorname{round}\left( 2^{M + 1} \cdot
       \operatorname{midpoint} \left(-\frac{-(k + 1)2^{-M}}{1 + k 2^{-M}},
       \frac{-k2^{-M}}{1 + (k + 1) 2^{-M}}\right) \right) \\
    &= 2^{-M - 1} \operatorname{round}\left( 2^{M + 1} \cdot \frac{1}{2} \left(
       \frac{-(k + 1)2^{-M}}{1 + k 2^{-M}} + \frac{-k2^{-M}}{1 + (k + 1) 2^{-M}}
       \right) \right) \\
    &= -2^{-M - 1} \operatorname{round}\left( \frac{
       (2k + 1) + (2k^2 + 2k + 1) 2^{-M} }
       {(1 + k 2^{-N})(1 + (k + 1) 2^{-N})} \right)  \quad\quad \text{(S3')}

The corresponding range and formula for `r = 1 + s` are:

.. math::
  \frac{1 - 2^{-M}}{1 + k 2^{-M}} < r \leq \frac{1 + 2^{-M}}{1 + (k + 1) 2^{-M}}

.. math::
  r &= 2^{-M-1} \operatorname{round}\left( 2^{M + 1} \cdot
       \operatorname{midpoint}\left( \frac{1 - 2^{-M}}{1 + k 2^{-M}},
          \frac{1 + 2^{-M}}{1 + (k + 1) 2^{-M}}\right) \right) \\
    &= 2^{-M-1} \operatorname{round}\left( 2^{M + 1} \cdot \frac{1}{2} \left(
       \frac{1 + 2^{-M}}{1 + (k + 1) 2^{-M}} + \frac{1 - 2^{-M}}{1 + k 2^{-M}}
    \right) \right) \\
    &= 2^{-M - 1} \operatorname{round}\left( 2^{M + 1} \cdot \frac{
       1 + \left(k + \frac{1}{2} \right) 2^{-M} - 2^{-2M-1} }{(1 + k 2^{-M})
       (1 + (k + 1) 2^{-M})} \right)

Edge cases
----------

1. When `k = 0`, notice that:

.. math::
  0 = k 2^{-N} \leq m_x < (k + 1) 2^{-N} = 2^{-N} = C,

so we can simply choose `r = 1` so that `\log(r) = 0` is exact, then `u = m_x`.
This will help reduce the accumulated errors when `m_x` is close to 0 while
maintaining the range reduction output's requirements.

2. When `k = 2^{N} - 1`, `\text{(S2)}` becomes:

.. math::
  -\frac{1}{2} - \frac{C - 2^{-M-1}}{2 - 2^{-M}} <> s \leq
  -\frac{1}{2} + \frac{C}{2}.

so when `C > 2^{-M - 1}` is a power of 2, we can always choose:

.. math::
  s = -\frac{1}{2}, \quad \text{i.e.} \quad r = \frac{1}{2}.

This reduction works well to avoid catastrophic cancellation happening when
`e_x = -1`.

This also works when `C = 2^{-M - 1}` if we relax the condition on `u` to
`|u| \leq C = 2^{-M-1}`.

Intermediate precision, and Ziv's test
--------------------------------------

In the fast phase, we want extra precision while performant, so we use
double-double precision for most intermediate computation steps, and employ Ziv
test to see if the result is accurate or not.  In our case, the Ziv's test can
be described as follow:

1. Let `re = re.hi + re.lo` be the double-double output of the fast phase
   computation.
2. Let `err` be an estimated upper bound of the errors of `re`.
3. If `\circ(re.hi + (re.lo - err)) == \circ(re.hi + (r.lo + err))` then the
   result is correctly rounded to double precision for the current rounding mode
   `\circ`.  Otherwise, the accurate phase with extra precision is needed.

For an easy and cheap estimation of the error bound `err`, since the range
reduction step described above is accurate, the errors of the result:

.. math::
  \log(x) &= e_x \log(2) - \log(r) + \log(1 + u) \\
          &\approx e_x \log(2) - \log(r) + u P(u)

come from 2 parts:

1. the look-up part: `e_x \log(2) - \log(r)`
2. the polynomial approximation part: `u P(u)`

The errors of the first part can be computed with a single `\operatorname{fma}`
operation:

.. math::
  err_1 = \operatorname{fma}(e_x, err(\log(2)), err(\log(r))),

and then combining with the errors of the second part for another
`\operatorname{fma}` operation:

.. math::
  err = \operatorname{fma}(u, err(P), err_1)


Accurate phase
==============

Extending range reduction
-------------------------

Since the output `u = r(1 + m_x) - 1` of the fast phase's range reduction
is computed exactly, we can apply further range reduction steps by
using the following formula:

.. math::
  u_{i + 1} = r_i(1 + u_i) - 1 = u_i \cdot r_i + (r_i - 1),

where `|u_i| < 2^{-N_i}` and `u_0 = u` is representable in double precision.

Let `s_i = r_i - 1`, then we can rewrite it as:

.. math::
  u_{i + 1} &= (1 + s_i)(1 + u_i) - 1 \\
            &= s_i u_i + u_i + s_i \\
            &= u_i (1 + s_i) + s_i
            &= s_i (1 + u_i) + u_i.

Then the bound on `u_{i + 1}` is translated to `s_i` as:

.. math::
  \frac{-2^{-N_{i + 1}} - u_i}{1 + u_i} < s_i < \frac{2^{-N_{i + 1}} - u_i}{1 + u_i}.

Let say we divide the interval `[0, 2^-{N_i})` into `2^{M_i}` subintervals
evenly and use the index `k` such that:

.. math::
  k 2^{-N_i - M_i} \leq u_i < (k + 1) 2^{-N_i - M_i},

to look-up for the reduction constant `s_{i, k}`.  In other word, `k` is given
by the formula:

.. math::
  k = \left\lfloor 2^{N_i + M_i} u_i \right\rfloor 

Notice that our reduction constant `s_{i, k}` must work for all `u_i` in the
interval `I = \{ v: k 2^{-N_i - M_i} \leq v < (k + 1) 2^{-N_i - M_i} \}`,
so it is bounded by:

.. math::
  \sup_{v \in I} \frac{-2^{-N_{i + 1}} - v}{1 + v} < s_{i, k} < \inf_{v \in I} \frac{2^{-N_{i + 1}} - v}{1 + v}

For a fixed constant `|C| < 1`, let `f(v) = \frac{C - v}{1 + v}`, then its derivative
is:

.. math::
  f'(v) = \frac{(-1)(1 + v) - (1)(C - v)}{(1 + v)^2} = \frac{-1 - C}{(1 + v)^2}.

Since `|C| < 1`, `f'(v) < 0` for all `v \neq -1`, so:

.. math::
  \sup_{v \in I} f(v) &= f \left( \inf\{ v: v \in I \} \right)
                       = f \left( k 2^{-N_i - M_i} \right) \\
  \inf_{v \in I} f(v) &= f \left( \sup\{ v: v \in I \} \right)
                       = f \left( (k + 1) 2^{-N_i - M_i} \right)

Hence we have the following bound on `s_{i, k}`:

.. math::
  \frac{-2^{-N_{i + 1}} - k 2^{-N_i - M_i}}{1 + k 2^{-N_i - M_i}} < s_{i, k}
  \leq \frac{2^{-N_{i + 1}} - (k + 1) 2^{-N_i - M_i}}{1 + (k + 1) 2^{-N_i - M_i}}

This interval is of length:

.. math::
  l &= \frac{2^{-N_{i + 1}} - (k + 1) 2^{-N_i - M_i}}{1 + (k + 1) 2^{-N_i - M_i}} -
  \frac{-2^{-N_{i + 1}} - k 2^{-N_i - M_i}}{1 + k 2^{-N_i - M_i}} \\
  &= \frac{2^{-N_{i + 1} + 1} - 2^{-N_i - M_i} + (2k + 1) 2^{-N_{i + 1} - N_i - M_i}}
      {(1 + k 2^{-N_i - M_i})(1 + (k + 1) 2^{-N_i -M_i})}

So in order to be able to find `s_{i, k}`, we need that:

.. math::
  2^{-N_{i + 1} + 1} - 2^{-N_i - M_i} + (2k + 1) 2^{-N_{i + 1} - N_i - M_i} > 0

This give us the following bound on `N_{i + 1}`:

.. math::
  N_{i + 1} \leq N_i + M_i + 1.

To make the range reduction effective, we will want to maximize `N_{i + 1}`, so
let consider the two cases: `N_{i + 1} = N_i + M_i + 1` and
`N_{i + 1} = N_i + M_i`.



The optimal choice to balance between maximizing `N_{i + 1}` and minimizing the
precision needed for `s_{i, k}` is:

.. math::
  N_{i + 1} = N_i + M_i,

and in this case, the optimal `\operatorname{ulp}(s_{i, k})` is:

.. math::
  \operatorname{ulp}(s_{i, k}) = 2^{-N_i - M_i}

and the corresponding `\operatorname{ulp}(u_{i + 1})` is:

.. math::
  \operatorname{ulp}(u_{i + 1}) &= \operatorname{ulp}(u_i) \operatorname{ulp}(s_{i, k}) \\
    &= \operatorname{ulp}(u_i) \cdot 2^{-N_i - M_i} \\
    &= \operatorname{ulp}(u_0) \cdot 2^{-N_0 - M_0} \cdot 2^{-N_0 - M_0 - M_1} \cdots 2^{-N_0 - M_0 - M_1 - \cdots - M_i} \\
    &= 2^{-N_0 - 53} \cdot 2^{-N_0 - M_0} \cdot 2^{-N_0 - M_0 - M_1} \cdots 2^{-N_0 - M_0 - M_1 - \cdots - M_i}

Since `|u_{i + 1}| < 2^{-N_{i + 1}} = 2^{-N_0 - M_1 - ... -M_i}`, the precision
of `u_{i + 1}` is:

.. math::
  \operatorname{prec}(u_{i + 1}) &= (N_0 + 53) + (N_0 + M_0) + \cdots +
    (N_0 + M_0 + \cdots + M_i) - (N_0 + M_0 + \cdots + M_i) \\
    &= (i + 1) N_0 + i M_0 + (i - 1) M_1 + \cdots + M_{i - 1} + 53

If we choose to have the same `M_0 = M_1 = \cdots = M_i = M`, this can be
simplified to:

.. math::
  \operatorname{prec}(u_{i + 1}) = (i + 1) N_0 + \frac{i(i + 1)}{2} \cdot M + 53.

We summarize the precision analysis for extending the range reduction in the
table below:

+-------+-----+-----------+------------+--------------+-----------------+-------------------+
| `N_0` | `M` | No. steps | Table size | Output bound | ulp(`s_{i, k}`) | prec(`u_{i + 1}`) |
+-------+-----+-----------+------------+--------------+-----------------+-------------------+
| 7     |  4  |         1 |         32 | `2^{-11}`    | `2^{-12}`       |  60               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         2 |         64 | `2^{-15}`    | `2^{-16}`       |  71               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         3 |         96 | `2^{-19}`    | `2^{-20}`       |  86               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         4 |        128 | `2^{-23}`    | `2^{-24}`       | 105               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         5 |        160 | `2^{-27}`    | `2^{-28}`       | 128               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         6 |        192 | `2^{-31}`    | `2^{-32}`       | 155               |
|       +-----+-----------+------------+--------------+-----------------+-------------------+
|       |  5  |         3 |        192 | `2^{-22}`    | `2^{-23}`       |  89               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         4 |        256 | `2^{-27}`    | `2^{-28}`       | 111               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         5 |        320 | `2^{-32}`    | `2^{-33}`       | 138               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         6 |        384 | `2^{-37}`    | `2^{-38}`       | 170               |
|       +-----+-----------+------------+--------------+-----------------+-------------------+
|       |  6  |         3 |        384 | `2^{-25}`    | `2^{-26}`       |  92               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         4 |        512 | `2^{-31}`    | `2^{-32}`       | 117               |
|       +-----+-----------+------------+--------------+-----------------+-------------------+
|       |  7  |         1 |        256 | `2^{-24}`    | `2^{-15}`       |  60               |
|       |     +-----------+------------+--------------+-----------------+-------------------+
|       |     |         2 |        512 | `2^{-21}`    | `2^{-22}`       |  74               |
+-------+-----+-----------+------------+--------------+-----------------+-------------------+

where:

- Number of steps = `i + 1`
- Table size = `(i + 1) 2^{M + 1}`
- Output bound = `2^{-N_{i + 1}} = 2^{-N_0 - (i + 1) M}`
- `\operatorname{ulp}(s_{i, k}) = 2^{-N_{i + 1} - 1}`
- `\operatorname{prec}(u_{i + 1}) = (i + 1) N_0 + \frac{i(i + 1)}{2} \cdot M + 53`
