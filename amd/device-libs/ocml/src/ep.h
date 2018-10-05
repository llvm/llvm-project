
#define ATTR __attribute__((const, overloadable))

#if defined FLOAT_SPECIALIZATION
#define T float
#define T2 float2
#define FMA BUILTIN_FMA_F32
#define RCP MATH_FAST_RCP
#define DIV(X,Y) MATH_FAST_DIV(X,Y)
#define LDEXP BUILTIN_FLDEXP_F32
#define SQRT MATH_FAST_SQRT
#define ISINF(X) BUILTIN_ISINF_F32(X)
#define USE_FMA HAVE_FAST_FMA32()
#define HIGH(X) AS_FLOAT(AS_UINT(X) & 0xfffff000U)
#define COPYSIGN BUILTIN_COPYSIGN_F64
#endif

#if defined DOUBLE_SPECIALIZATION
#define T double
#define T2 double2
#define FMA BUILTIN_FMA_F64
#define RCP MATH_FAST_RCP
#define DIV(X,Y) MATH_FAST_DIV(X,Y)
#define LDEXP BUILTIN_FLDEXP_F64
#define SQRT MATH_FAST_SQRT
#define ISINF(X) BUILTIN_ISINF_F64(X)
#define USE_FMA true
#define HIGH(X) AS_DOUBLE(AS_ULONG(X) & 0xfffffffff8000000UL)
#define COPYSIGN BUILTIN_COPYSIGN_F32
#endif

#if defined HALF_SPECIALIZATION
#define T half
#define T2 half2
#define FMA BUILTIN_FMA_F16
#define RCP MATH_FAST_RCP
#define DIV(X,Y) MATH_FAST_DIV(X,Y)
#define LDEXP BUILTIN_FLDEXP_F16
#define SQRT MATH_FAST_SQRT
#define ISINF(X) BUILTIN_ISINF_F16(X)
#define USE_FMA true
#define HIGH(X) AS_HALF(AS_USHORT(X) & (ushort)0xffc0U)
#define COPYSIGN BUILTIN_COPYSIGN_F16
#endif

static ATTR T2
con(T a, T b)
{
    return (T2)(b, a);
}

static ATTR T2
csgn(T2 a, T b)
{
    return con(COPYSIGN(a.hi, b), COPYSIGN(a.lo, b));
}

static ATTR T2
csgn(T2 a, T2 b)
{
    return con(COPYSIGN(a.hi, b.hi), COPYSIGN(a.lo, b.lo));
}

static ATTR T2
fadd(T a, T b)
{
    T s = a + b;
    return con(s, b - (s - a));
}

static ATTR T2
nrm(T2 a)
{
    return fadd(a.hi, a.lo);
}

static ATTR T2
onrm(T2 a)
{
    T s = a.hi + a.lo;
    T t = a.lo - (s - a.hi);
    s = ISINF(a.hi) ? a.hi : s;
    return con(s, ISINF(s) ? (T)0 : t);
}

static ATTR T2
fsub(T a, T b)
{
    T d = a - b;
    return con(d, (a - d) - b);
}

static ATTR T2
add(T a, T b)
{
    T s = a + b;
    T d = s - a;
    return con(s, (a - (s - d)) + (b - d));
}

static ATTR T2
sub(T a, T b)
{
    T d = a - b;
    T e = d - a;
    return con(d, (a - (d - e)) - (b + e));
}

static ATTR T2
mul(T a, T b)
{
    T p = a * b;
    if (USE_FMA) {
        return con(p, FMA(a, b, -p));
    } else {
        T ah = HIGH(a);
        T al = a - ah;
        T bh = HIGH(b);
        T bl = b - bh;
        T p = a * b;
        return con(p, ((ah*bh - p) + ah*bl + al*bh) + al*bl);
    }
}

static ATTR T2
sqr(T a)
{
    T p = a * a;
    if (USE_FMA) {
        return con(p, FMA(a, a, -p));
    } else {
        T ah = HIGH(a);
        T al = a - ah;
        return con(p, ((ah*ah - p) + 2.0f*ah*al) + al*al);
    }
}

static ATTR T2
add(T2 a, T b)
{
    T2 s = add(a.hi, b);
    s.lo += a.lo;
    return nrm(s);
}

static ATTR T2
fadd(T2 a, T b)
{
    T2 s = fadd(a.hi, b);
    s.lo += a.lo;
    return nrm(s);
}

static ATTR T2
add(T a, T2 b)
{
    T2 s = add(a, b.hi);
    s.lo += b.lo;
    return nrm(s);
}

static ATTR T2
fadd(T a, T2 b)
{
    T2 s = fadd(a, b.hi);
    s.lo += b.lo;
    return nrm(s);
}

static ATTR T2
add(T2 a, T2 b)
{
    T2 s = add(a.hi, b.hi);
    T2 t = add(a.lo, b.lo);
    s.lo += t.hi;
    s = nrm(s);
    s.lo += t.lo;
    return nrm(s);
}

static ATTR T2
fadd(T2 a, T2 b)
{
    T2 s = fadd(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return nrm(s);
}

static ATTR T2
sub(T2 a, T b)
{
    T2 d = sub(a.hi, b);
    d.lo += a.lo;
    return nrm(d);
}

static ATTR T2
fsub(T2 a, T b)
{
    T2 d = fsub(a.hi, b);
    d.lo += a.lo;
    return nrm(d);
}

static ATTR T2
sub(T a, T2 b)
{
    T2 d = sub(a, b.hi);
    d.lo -= b.lo;
    return nrm(d);
}

static ATTR T2
fsub(T a, T2 b)
{
    T2 d = fsub(a, b.hi);
    d.lo -= b.lo;
    return nrm(d);
}

static ATTR T2
sub(T2 a, T2 b)
{
    T2 d = sub(a.hi, b.hi);
    T2 e = sub(a.lo, b.lo);
    d.lo += e.hi;
    d = nrm(d);
    d.lo += e.lo;
    return nrm(d);
}

static ATTR T2
fsub(T2 a, T2 b)
{
    T2 d = fsub(a.hi, b.hi);
    d.lo = d.lo + a.lo - b.lo;
    return nrm(d);
}

static ATTR T2
ldx(T2 a, int e)
{
    return con(LDEXP(a.hi, e), LDEXP(a.lo, e));
}

static ATTR T2
mul(T2 a, T b)
{
    T2 p = mul(a.hi, b);
    if (USE_FMA) {
        p.lo = FMA(a.lo, b, p.lo);
    } else {
        p.lo += a.lo * b;
    }
    return nrm(p);
}

static ATTR T2
omul(T2 a, T b)
{
    T2 p = mul(a.hi, b);
    if (USE_FMA) {
        p.lo = FMA(a.lo, b, p.lo);
    } else {
        p.lo += a.lo * b;
    }
    return onrm(p);
}

static ATTR T2
mul(T a, T2 b)
{
    T2 p = mul(a, b.hi);
    if (USE_FMA) {
        p.lo = FMA(a, b.lo, p.lo);
    } else {
        p.lo += a * b.lo;
    }
    return nrm(p);
}

static ATTR T2
omul(T a, T2 b)
{
    T2 p = mul(a, b.hi);
    if (USE_FMA) {
        p.lo = FMA(a, b.lo, p.lo);
    } else {
        p.lo += a * b.lo;
    }
    return onrm(p);
}

static ATTR T2
mul(T2 a, T2 b)
{
    T2 p = mul(a.hi, b.hi);
    if (USE_FMA) {
        p.lo += FMA(a.hi, b.lo, a.lo*b.hi);
    } else {
        p.lo += a.hi*b.lo + a.lo*b.hi;
    }
    return nrm(p);
}

static ATTR T2
omul(T2 a, T2 b)
{
    T2 p = mul(a.hi, b.hi);
    if (USE_FMA) {
        p.lo += FMA(a.hi, b.lo, a.lo*b.hi);
    } else {
        p.lo += a.hi*b.lo + a.lo*b.hi;
    }
    return onrm(p);
}

static ATTR T2
div(T a, T b)
{
    T r = RCP(b);
    T qhi = a * r;
    T2 p = mul(qhi, b);
    T2 d = fsub(a, p.hi);
    d.lo -= p.lo;
    T qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

static ATTR T2
div(T2 a, T b)
{
    T r = RCP(b);
    T qhi = a.hi * r;
    T2 p = mul(qhi, b);
    T2 d = fsub(a.hi, p.hi);
    d.lo = d.lo + a.lo - p.lo;
    T qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

static ATTR T2
div(T a, T2 b)
{
    T r = RCP(b.hi);
    T qhi = a * r;
    T2 p = mul(qhi, b);
    T2 d = fsub(a, p.hi);
    d.lo -= p.lo;
    T qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

static ATTR T2
fdiv(T2 a, T2 b)
{
    T r = RCP(b.hi);
    T qhi = a.hi * r;
    T2 p = mul(qhi, b);
    T2 d = fsub(a.hi, p.hi);
    d.lo = d.lo - p.lo + a.lo;
    T qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

static ATTR T2
div(T2 a, T2 b)
{
    T y = RCP(b.hi);
    T qhi = a.hi * y;
    T2 r = fsub(a, mul(qhi, b));
    T qmi = r.hi * y;
    r = fsub(r, mul(qmi, b));
    T qlo = r.hi * y;
    T2 q = fadd(qhi, qmi);
    q.lo += qlo;
    return nrm(q);
}

static ATTR T2
rcp(T b)
{
    T qhi = RCP(b);
    T2 p = mul(qhi, b);
    T2 d = fsub((T)1, p.hi);
    d.lo -= p.lo;
    T qlo = (d.hi + d.lo) * qhi;
    return fadd(qhi, qlo);
}

static ATTR T2
frcp(T2 b)
{
    T qhi = RCP(b.hi);
    T2 p = mul(qhi, b);
    T2 d = fsub((T)1, p.hi);
    d.lo -= p.lo;
    T qlo = (d.hi + d.lo) * qhi;
    return fadd(qhi, qlo);
}

static ATTR T2
rcp(T2 b)
{
    T qhi = RCP(b.hi);
    T2 r = fsub((T)1, mul(qhi, b));
    T qmi = r.hi * qhi;
    r = fsub(r, mul(qmi, b));
    T qlo = r.hi * qhi;
    T2 q = fadd(qhi, qmi);
    q.lo += qlo;
    return nrm(q);
}

static ATTR T2
sqr(T2 a)
{
    T2 p = sqr(a.hi);
    if (USE_FMA) {
        p.lo = FMA(a.lo, a.lo, FMA(a.hi, (T)2*a.lo, p.lo));
    } else {
        p.lo = p.lo + a.hi * a.lo * (T)2 + a.lo * a.lo;
    }
    return fadd(p.hi, p.lo);
}

static ATTR T2
root2(T a)
{
    T shi = SQRT(a);
    T2 e = fsub(a, sqr(shi));
    T slo = DIV(e.hi, (T)2 * shi);
    return fadd(shi, slo);
}

static ATTR T2
root2(T2 a)
{
    T shi = SQRT(a.hi);
    T2 e = fsub(a, sqr(shi));
    T slo = DIV(e.hi, (T)2 * shi);
    return fadd(shi, slo);
}

#undef ATTR
#undef T
#undef T2
#undef FMA
#undef RCP
#undef DIV
#undef LDEXP
#undef SQRT
#undef ISINF
#undef USE_FMA
#undef HIGH
#undef COPYSIGN

