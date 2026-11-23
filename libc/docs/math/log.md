(log_algorithm)=

# Log/Log10/Log2 Algorithm

In this short note, we will discuss in detail about the computation of
{math}`\log(x)` function, with double precision inputs, in particular, the range
reduction steps and error analysis. The algorithm is broken down into 2 main
phases as follow:

1. Fast phase:

   1. Range reduction
   2. Polynomial approximation
   3. Ziv's test

2. Accurate phase (if Ziv's test failed):

   1. Further range reduction
   2. Polynomial approximation

## Fast phase

### Range reduction

Let {math}`x = 2^{e_x} (1 + m_x)` be a normalized double precision number, in which
{math}`-1074 \leq e_x \leq 1022` and {math}`0 \leq m_x < 1` such that
{math}`2^{52} m_x \in \mathbb{Z}`.

Then from the properties of logarithm:

```{math}
\log(x) &= \log\left( 2^{e_x} (1 + m_x) \right) \\
        &= \log\left( 2^{e_x} \right) + \log(1 + m_x) \\
        &= e_x \log(2) + \log(1 + m_x)
```

the computation of {math}`\log(x)` can be reduced to:

1. compute the product of {math}`e_x` and {math}`\log(2)`,
2. compute {math}`\log(1 + m_x)` for {math}`0 \leq m_x < 1`,
3. add step 1 and 2.

To compute {math}`\log(1 + m_x)` in step 2, we can reduce the range further by finding
{math}`r > 0` such that:

```{math}
| r(1 + m_x) - 1 | < C \quad \quad \text{(R1)}
```

for small {math}`0 < C < 1`. Then if we let {math}`u = r(1 + m_x) - 1`, {math}`|u| < C`:

```{math}
\log(1 + m_x) &= \log \left( \frac{r (1 + m_x)}{r} \right) \\
              &= \log(r (1 + m_x) ) - \log(r) \\
              &= \log(1 + u) - \log(r)
```

and step 2 can be computed with:

1. extract {math}`r` and {math}`-\log(r)` from look-up tables,
2. compute the reduced argument {math}`u = r(1 + m_x) - 1`,
3. compute {math}`\log(1 + u)` by polynomial approximation or further range reduction,
4. add the results of items 1 and 3.

### How to derive {math}`r`

For an efficient implementation, we would like to use the first {math}`M` significant
bits of {math}`m_x` to look up for {math}`r`. In particular, we would like to find a value
of {math}`r` that works for all {math}`m_x` satisfying:

```{math}
k 2^{-M} \leq m_x < (k + 1) 2^{-M} \quad \text{for some} \quad
k = 0..2^{M} - 1. \quad\quad \text{(M1)}
```

Let {math}`r = 1 + s`, then {math}`u` can be expressed in terms of {math}`s` as:

```{math}
u &= r(1 + m_x) - 1 \\
  &= (1 + s)(1 + m_x) - 1 \\
  &= s m_x + s + m_x  &\quad\quad \text{(U1)} \\
  &= s (1 + m_x) + m_x \\
  &= m_x (1 + s) + s.
```

From the condition {math}`\text{(R1)}`, {math}`s` is bounded by:

```{math}
\frac{-C - m_x}{1 + m_x} < s < \frac{C - m_x}{1 + m_x} \quad\quad \text{(S1)}.
```

Since our reduction constant {math}`s` must work for all {math}`m_x` in the interval
{math}`I = \{ v: k 2^{-M} \leq v < (k + 1) 2^{-M} \}`, {math}`s` is bounded by:

```{math}
\sup_{v \in I} \frac{-C - v}{1 + v} < s < \inf_{v \in I} \frac{C - v}{1 + v}
```

For a fixed constant {math}`|c| < 1`, let {math}`f(v) = \frac{c - v}{1 + v}`, then its
derivative is:

```{math}
f'(v) = \frac{(-1)(1 + v) - (1)(c - v)}{(1 + v)^2} = \frac{-1 - c}{(1 + v)^2}.
```

Since {math}`|c| < 1`, {math}`f'(v) < 0` for all {math}`v \neq -1`, so:

```{math}
\sup_{v \in I} f(v) &= f \left( \inf\{ v: v \in I \} \right)
                     = f \left( k 2^{-M} \right) \\
\inf_{v \in I} f(v) &= f \left( \sup\{ v: v \in I \} \right)
                     = f \left( (k + 1) 2^{-M} \right)
```

Hence we have the following bound on {math}`s`:

```{math}
\frac{-C - k 2^{-M}}{1 + k 2^{-M}} < s \leq
\frac{C - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}}. \quad\quad \text{(S2)}
```

In order for {math}`s` to exist, we need that:

```{math}
\frac{C - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}} >
\frac{-C - k 2^{-M}}{1 + k 2^{-M}}
```

which is equivalent to:

```{math}
\quad\quad 2C - 2^{-M} + (2k + 1) 2^{-M} C > 0
\iff C > \frac{2^{-M - 1}}{1 + (2k + 1) 2^{-M - 1}} \quad\quad \text{(C1)}.
```

Consider the case {math}`C = 2^{-N}`. Since {math}`0 \leq k \leq 2^M - 1,` the right hand
side of {math}`\text{(C1)}` is bounded by:

```{math}
2^{-M - 1} > \frac{2^{-M - 1}}{1 + (2k + 1) 2^{-M - 1}} \geq
\frac{2^{-M - 1}}{1 + (2^{M + 1} - 1) 2^{-M - 1}} > 2^{-M - 2}.
```

Hence, from {math}`\text{(C1)}`, being an exact power of 2, {math}`C = 2^{-N}` is bounded below
by:

```{math}
C = 2^{-N} \geq 2^{-M - 1}.
```

To make the range reduction efficient, we will want to minimize {math}`C` (maximize
{math}`N`) while keeping the required precision of {math}`s`({math}`r`) as low as possible. And
for that, we will consider the following two cases: {math}`N = M + 1` and {math}`N = M`.

#### Case 1 - {math}`N = M + 1`

When {math}`N = M + 1`, {math}`\text{(S2)}` becomes:

```{math}
\frac{-2^{-M - 1} - k 2^{-M}}{1 + k 2^{-M}} < s <
\frac{2^{-M - 1} - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}}.
\quad\quad \text{(S2')}
```

This is an interval of length:

```{math}
l &= \frac{2^{-M - 1} - (k + 1) 2^{-M}}{1 + (k + 1) 2^{-M}} -
     \frac{-2^{-M - 1} - k 2^{-M}}{1 + k 2^{-M}} \\
  &= \frac{(2k + 1)2^{-2M - 1}}{(1 + k 2^{-M})(1 + (k + 1)2^{-M})}
  \quad\quad \text{(L1)}
```

As a function of {math}`k`, the length {math}`l` has its derivative with respect to {math}`k`:

```{math}
\frac{dl}{dk} =
\frac{2^{2M + 1} - 2k(k + 1) - 1}
     {2^{4M}(1 + k 2^{-M})^2 (1 + (k + 1) 2^{-M})^2}
```

which is always positive for {math}`0 \leq k \leq 2^M - 1`. So for all
{math}`0 < k < 2^{-M}` ({math}`k = 0` will be treated differently in edge cases), and for
{math}`M > 2`, {math}`l` is bounded below by:

```{math}
l > 2^{-2M}.
```

It implies that we can always find {math}`s` with {math}`\operatorname{ulp}(s) = 2^{-2M}`.
And from {math}`\text{(U1)}`, {math}`u = s(1 + m_x) + m_x`, its {math}`ulp` is:

```{math}
\operatorname{ulp}(u) &= \operatorname{ulp}(s) \cdot \operatorname{ulp}(m_x) \\
                      &= 2^{-2M} \operatorname{ulp}(m_x).
```

Since:

```{math}
|u| < C = 2^{-N} = 2^{-M - 1},
```

Its required precision is:

```{math}
\operatorname{prec}(u) &= \log_2(2^{-M-1} / \operatorname{ulp}(u)) \\
                       &= \log_2(2^{M - 1} / \operatorname{ulp}(m_x)) \\
                       &= M - 1 - \log_2(\operatorname{ulp}(m_x)).
```

This means that in this case, we cannot restrict {math}`u` to be exactly representable
in double precision for double precision input {math}`x` with {math}`M > 2`. Nonetheless,
for a reasonable value of {math}`M`, we can have {math}`u` exactly representable in double
precision for single precision input {math}`x` ({math}`\operatorname{ulp}(m_x) = 2^{-23}`)
such that {math}`|u| < 2^{-M - 1}` using a look-up table of size {math}`2^M`.

A particular formula for {math}`s` can be derived from {math}`\text{(S2')}` by the midpoint
formula:

```{math}
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
```

The corresponding range and formula for {math}`r = 1 + s` are:

```{math}
\frac{1 - 2^{-M - 1}}{1 + k 2^{-M}} < r \leq
\frac{1 + 2^{-M - 1}}{1 + (k + 1) 2^{-M}}
```

```{math}
r &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot
     \operatorname{midpoint}\left( \frac{1 - 2^{-M - 1}}{1 + k 2^{-M}},
        \frac{1 + 2^{-M - 1}}{1 + (k + 1) 2^{-M}}\right) \right) \\
  &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot \frac{1}{2} \left(
     \frac{1 + 2^{-M-1}}{1 + (k + 1) 2^{-M}} + \frac{1 - 2^{-M-1}}{1 + k 2^{-M}}
  \right) \right) \\
  &= 2^{-2M} \operatorname{round}\left( 2^{2M} \cdot \frac{
     1 + \left(k + \frac{1}{2} \right) 2^{-M} - 2^{-2M-2} }{(1 + k 2^{-M})
     (1 + (k + 1) 2^{-M})} \right)
```

#### Case 1 - {math}`N = M`

When {math}`N = M`, {math}`\text{(S2)}` becomes:

```{math}
\frac{-(k + 1)2^{-M}}{1 + k 2^{-M}} < s < \frac{-k 2^{-M}}{1 + (k + 1) 2^{-M}}
\quad\quad \text{(S2")}
```

This is an interval of length:

```{math}
l &= \frac{- k 2^{-M}}{1 + (k + 1) 2^{-M}} -
     \frac{- (k + 1) 2^{-M}}{1 + k 2^{-M}} \\
  &= \frac{2^{-M} (1 + (2k + 1) 2^{-M})}{(1 + k 2^{-M})(1 + (k + 1)2^{-M})}
  \quad\quad \text{(L1')}
```

As a function of {math}`k`, its derivative with respect to {math}`k`:

```{math}
\frac{dl}{dk} =
-\frac{2^{-2M}(k(k + 1)2^{-M + 1} + 2^{-M} + 2k + 1)}
      {(1 + k 2^{-M})^2 (1 + (k + 1) 2^{-M})^2}
```

which is always negative for {math}`0 \leq k \leq 2^M - 1`. So for {math}`M > 1`, {math}`l` is
bounded below by:

```{math}
l > \frac{2^{-M - 1} (3 - 2^{-M})}{2 - 2^{-M}} > 2^{-M - 1}.
```

It implies that we can always find {math}`s` with {math}`\operatorname{ulp}(s) = 2^{-M-1}`.
And from {math}`\text{(U1)}`, {math}`u = s(1 + m_x) + m_x`, its {math}`ulp` is:

```{math}
\operatorname{ulp}(u) &= \operatorname{ulp}(s) \cdot \operatorname{ulp}(m_x) \\
                      &= 2^{-M - 1} \operatorname{ulp}(m_x).
```

Since:

```{math}
|u| < C = 2^{-N} = 2^{-M},
```

Its required precision is:

```{math}
\operatorname{prec}(u) &= \log_2(2^{-M} / \operatorname{ulp}(u)) \\
                       &= \log_2(2 / \operatorname{ulp}(m_x)) \\
                       &= 1 - \log_2(\operatorname{ulp}(m_x)).
```

Hence, for double precision {math}`x`, {math}`\operatorname{ulp}(m_x) = 2^{-52}`, and the
precision needed for {math}`u` is {math}`\operatorname{prec}(u) = 53`, i.e., {math}`u` can be
exactly representable in double precision. And in this case, {math}`s` can be
derived from {math}`\text{(S2")}` by the midpoint formula:

```{math}
s &= 2^{-M - 1} \operatorname{round}\left( 2^{M + 1} \cdot
     \operatorname{midpoint} \left(-\frac{-(k + 1)2^{-M}}{1 + k 2^{-M}},
     \frac{-k2^{-M}}{1 + (k + 1) 2^{-M}}\right) \right) \\
  &= 2^{-M - 1} \operatorname{round}\left( 2^{M + 1} \cdot \frac{1}{2} \left(
     \frac{-(k + 1)2^{-M}}{1 + k 2^{-M}} + \frac{-k2^{-M}}{1 + (k + 1) 2^{-M}}
     \right) \right) \\
  &= -2^{-M - 1} \operatorname{round}\left( \frac{
     (2k + 1) + (2k^2 + 2k + 1) 2^{-M} }
     {(1 + k 2^{-N})(1 + (k + 1) 2^{-N})} \right)  \quad\quad \text{(S3')}
```

The corresponding range and formula for {math}`r = 1 + s` are:

```{math}
\frac{1 - 2^{-M}}{1 + k 2^{-M}} < r \leq \frac{1 + 2^{-M}}{1 + (k + 1) 2^{-M}}
```

```{math}
r &= 2^{-M-1} \operatorname{round}\left( 2^{M + 1} \cdot
     \operatorname{midpoint}\left( \frac{1 - 2^{-M}}{1 + k 2^{-M}},
        \frac{1 + 2^{-M}}{1 + (k + 1) 2^{-M}}\right) \right) \\
  &= 2^{-M-1} \operatorname{round}\left( 2^{M + 1} \cdot \frac{1}{2} \left(
     \frac{1 + 2^{-M}}{1 + (k + 1) 2^{-M}} + \frac{1 - 2^{-M}}{1 + k 2^{-M}}
  \right) \right) \\
  &= 2^{-M - 1} \operatorname{round}\left( 2^{M + 1} \cdot \frac{
     1 + \left(k + \frac{1}{2} \right) 2^{-M} - 2^{-2M-1} }{(1 + k 2^{-M})
     (1 + (k + 1) 2^{-M})} \right)
```

### Edge cases

1. When {math}`k = 0`, notice that:

   ```{math}
   0 = k 2^{-N} \leq m_x < (k + 1) 2^{-N} = 2^{-N} = C,
   ```

   so we can simply choose {math}`r = 1` so that {math}`\log(r) = 0` is exact, then {math}`u = m_x`.
   This will help reduce the accumulated errors when {math}`m_x` is close to 0 while
   maintaining the range reduction output's requirements.

2. When {math}`k = 2^{N} - 1`, {math}`\text{(S2)}` becomes:

   ```{math}
   -\frac{1}{2} - \frac{C - 2^{-M-1}}{2 - 2^{-M}} <> s \leq
   -\frac{1}{2} + \frac{C}{2}.
   ```

   so when {math}`C > 2^{-M - 1}` is a power of 2, we can always choose:

   ```{math}
   s = -\frac{1}{2}, \quad \text{i.e.} \quad r = \frac{1}{2}.
   ```

   This reduction works well to avoid catastrophic cancellation happening when
   {math}`e_x = -1`.

   This also works when {math}`C = 2^{-M - 1}` if we relax the condition on {math}`u` to
   {math}`|u| \leq C = 2^{-M-1}`.

### Intermediate precision, and Ziv's test

In the fast phase, we want extra precision while performant, so we use
double-double precision for most intermediate computation steps, and employ Ziv
test to see if the result is accurate or not. In our case, the Ziv's test can
be described as follow:

1. Let {math}`re = re.hi + re.lo` be the double-double output of the fast phase
   computation.
2. Let {math}`err` be an estimated upper bound of the errors of {math}`re`.
3. If {math}`\circ(re.hi + (re.lo - err)) == \circ(re.hi + (r.lo + err))` then the
   result is correctly rounded to double precision for the current rounding mode
   {math}`\circ`. Otherwise, the accurate phase with extra precision is needed.

For an easy and cheap estimation of the error bound {math}`err`, since the range
reduction step described above is accurate, the errors of the result:

```{math}
\log(x) &= e_x \log(2) - \log(r) + \log(1 + u) \\
        &\approx e_x \log(2) - \log(r) + u P(u)
```

come from 2 parts:

1. the look-up part: {math}`e_x \log(2) - \log(r)`
2. the polynomial approximation part: {math}`u P(u)`

The errors of the first part can be computed with a single {math}`\operatorname{fma}`
operation:

```{math}
err_1 = \operatorname{fma}(e_x, err(\log(2)), err(\log(r))),
```

and then combining with the errors of the second part for another
{math}`\operatorname{fma}` operation:

```{math}
err = \operatorname{fma}(u, err(P), err_1)
```

## Accurate phase

### Extending range reduction

Since the output {math}`u = r(1 + m_x) - 1` of the fast phase's range reduction
is computed exactly, we can apply further range reduction steps by
using the following formula:

```{math}
u_{i + 1} = r_i(1 + u_i) - 1 = u_i \cdot r_i + (r_i - 1),
```

where {math}`|u_i| < 2^{-N_i}` and {math}`u_0 = u` is representable in double precision.

Let {math}`s_i = r_i - 1`, then we can rewrite it as:

```{math}
u_{i + 1} &= (1 + s_i)(1 + u_i) - 1 \\
          &= s_i u_i + u_i + s_i \\
          &= u_i (1 + s_i) + s_i
          &= s_i (1 + u_i) + u_i.
```

Then the bound on {math}`u_{i + 1}` is translated to {math}`s_i` as:

```{math}
\frac{-2^{-N_{i + 1}} - u_i}{1 + u_i} < s_i < \frac{2^{-N_{i + 1}} - u_i}{1 + u_i}.
```

Let say we divide the interval {math}`[0, 2^-{N_i})` into {math}`2^{M_i}` subintervals
evenly and use the index {math}`k` such that:

```{math}
k 2^{-N_i - M_i} \leq u_i < (k + 1) 2^{-N_i - M_i},
```

to look-up for the reduction constant {math}`s_{i, k}`. In other word, {math}`k` is given
by the formula:

```{math}
k = \left\lfloor 2^{N_i + M_i} u_i \right\rfloor
```

Notice that our reduction constant {math}`s_{i, k}` must work for all {math}`u_i` in the
interval {math}`I = \{ v: k 2^{-N_i - M_i} \leq v < (k + 1) 2^{-N_i - M_i} \}`,
so it is bounded by:

```{math}
\sup_{v \in I} \frac{-2^{-N_{i + 1}} - v}{1 + v} < s_{i, k} < \inf_{v \in I} \frac{2^{-N_{i + 1}} - v}{1 + v}
```

For a fixed constant {math}`|C| < 1`, let {math}`f(v) = \frac{C - v}{1 + v}`, then its derivative
is:

```{math}
f'(v) = \frac{(-1)(1 + v) - (1)(C - v)}{(1 + v)^2} = \frac{-1 - C}{(1 + v)^2}.
```

Since {math}`|C| < 1`, {math}`f'(v) < 0` for all {math}`v \neq -1`, so:

```{math}
\sup_{v \in I} f(v) &= f \left( \inf\{ v: v \in I \} \right)
                     = f \left( k 2^{-N_i - M_i} \right) \\
\inf_{v \in I} f(v) &= f \left( \sup\{ v: v \in I \} \right)
                     = f \left( (k + 1) 2^{-N_i - M_i} \right)
```

Hence we have the following bound on {math}`s_{i, k}`:

```{math}
\frac{-2^{-N_{i + 1}} - k 2^{-N_i - M_i}}{1 + k 2^{-N_i - M_i}} < s_{i, k}
\leq \frac{2^{-N_{i + 1}} - (k + 1) 2^{-N_i - M_i}}{1 + (k + 1) 2^{-N_i - M_i}}
```

This interval is of length:

```{math}
l &= \frac{2^{-N_{i + 1}} - (k + 1) 2^{-N_i - M_i}}{1 + (k + 1) 2^{-N_i - M_i}} -
\frac{-2^{-N_{i + 1}} - k 2^{-N_i - M_i}}{1 + k 2^{-N_i - M_i}} \\
&= \frac{2^{-N_{i + 1} + 1} - 2^{-N_i - M_i} + (2k + 1) 2^{-N_{i + 1} - N_i - M_i}}
    {(1 + k 2^{-N_i - M_i})(1 + (k + 1) 2^{-N_i -M_i})}
```

So in order to be able to find {math}`s_{i, k}`, we need that:

```{math}
2^{-N_{i + 1} + 1} - 2^{-N_i - M_i} + (2k + 1) 2^{-N_{i + 1} - N_i - M_i} > 0
```

This give us the following bound on {math}`N_{i + 1}`:

```{math}
N_{i + 1} \leq N_i + M_i + 1.
```

To make the range reduction effective, we will want to maximize {math}`N_{i + 1}`, so
let consider the two cases: {math}`N_{i + 1} = N_i + M_i + 1` and
{math}`N_{i + 1} = N_i + M_i`.

The optimal choice to balance between maximizing {math}`N_{i + 1}` and minimizing the
precision needed for {math}`s_{i, k}` is:

```{math}
N_{i + 1} = N_i + M_i,
```

and in this case, the optimal {math}`\operatorname{ulp}(s_{i, k})` is:

```{math}
\operatorname{ulp}(s_{i, k}) = 2^{-N_i - M_i}
```

and the corresponding {math}`\operatorname{ulp}(u_{i + 1})` is:

```{math}
\operatorname{ulp}(u_{i + 1}) &= \operatorname{ulp}(u_i) \operatorname{ulp}(s_{i, k}) \\
  &= \operatorname{ulp}(u_i) \cdot 2^{-N_i - M_i} \\
  &= \operatorname{ulp}(u_0) \cdot 2^{-N_0 - M_0} \cdot 2^{-N_0 - M_0 - M_1} \cdots 2^{-N_0 - M_0 - M_1 - \cdots - M_i} \\
  &= 2^{-N_0 - 53} \cdot 2^{-N_0 - M_0} \cdot 2^{-N_0 - M_0 - M_1} \cdots 2^{-N_0 - M_0 - M_1 - \cdots - M_i}
```

Since {math}`|u_{i + 1}| < 2^{-N_{i + 1}} = 2^{-N_0 - M_1 - ... -M_i}`, the precision
of {math}`u_{i + 1}` is:

```{math}
\operatorname{prec}(u_{i + 1}) &= (N_0 + 53) + (N_0 + M_0) + \cdots +
  (N_0 + M_0 + \cdots + M_i) - (N_0 + M_0 + \cdots + M_i) \\
  &= (i + 1) N_0 + i M_0 + (i - 1) M_1 + \cdots + M_{i - 1} + 53
```

If we choose to have the same {math}`M_0 = M_1 = \cdots = M_i = M`, this can be
simplified to:

```{math}
\operatorname{prec}(u_{i + 1}) = (i + 1) N_0 + \frac{i(i + 1)}{2} \cdot M + 53.
```

We summarize the precision analysis for extending the range reduction in the
table below:

```{eval-rst}
+-------------+-----------+-----------+------------+--------------------+-----------------------+-------------------------+
| :math:`N_0` | :math:`M` | No. steps | Table size | Output bound       | ulp(:math:`s_{i, k}`) | prec(:math:`u_{i + 1}`) |
+-------------+-----------+-----------+------------+--------------------+-----------------------+-------------------------+
| 7           |  4        |         1 |         32 | :math:`2^{-11}`    | :math:`2^{-12}`       |  60                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         2 |         64 | :math:`2^{-15}`    | :math:`2^{-16}`       |  71                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         3 |         96 | :math:`2^{-19}`    | :math:`2^{-20}`       |  86                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         4 |        128 | :math:`2^{-23}`    | :math:`2^{-24}`       | 105                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         5 |        160 | :math:`2^{-27}`    | :math:`2^{-28}`       | 128                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         6 |        192 | :math:`2^{-31}`    | :math:`2^{-32}`       | 155                     |
|             +-----------+-----------+------------+--------------------+-----------------------+-------------------------+
|             |  5        |         3 |        192 | :math:`2^{-22}`    | :math:`2^{-23}`       |  89                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         4 |        256 | :math:`2^{-27}`    | :math:`2^{-28}`       | 111                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         5 |        320 | :math:`2^{-32}`    | :math:`2^{-33}`       | 138                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         6 |        384 | :math:`2^{-37}`    | :math:`2^{-38}`       | 170                     |
|             +-----------+-----------+------------+--------------------+-----------------------+-------------------------+
|             |  6        |         3 |        384 | :math:`2^{-25}`    | :math:`2^{-26}`       |  92                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         4 |        512 | :math:`2^{-31}`    | :math:`2^{-32}`       | 117                     |
|             +-----------+-----------+------------+--------------------+-----------------------+-------------------------+
|             |  7        |         1 |        256 | :math:`2^{-24}`    | :math:`2^{-15}`       |  60                     |
|             |           +-----------+------------+--------------------+-----------------------+-------------------------+
|             |           |         2 |        512 | :math:`2^{-21}`    | :math:`2^{-22}`       |  74                     |
+-------------+-----------+-----------+------------+--------------------+-----------------------+-------------------------+
```

where:

- Number of steps = {math}`i + 1`
- Table size = {math}`(i + 1) 2^{M + 1}`
- Output bound = {math}`2^{-N_{i + 1}} = 2^{-N_0 - (i + 1) M}`
- {math}`\operatorname{ulp}(s_{i, k}) = 2^{-N_{i + 1} - 1}`
- {math}`\operatorname{prec}(u_{i + 1}) = (i + 1) N_0 + \frac{i(i + 1)}{2} \cdot M + 53`
