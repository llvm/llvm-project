// RUN: %clang_cc1 %s

int a, b, c, d, e, f, g, h, i, j, k, l;

void
f1 (void)
{
  __asm__ volatile (""
            : [a] "+r" (a), [b] "+r" (b), [c] "+r" (c), [d] "+r" (d),
              [e] "+r" (e), [f] "+r" (f), [g] "+r" (g), [h] "+r" (h),
              [i] "+r" (i), [j] "+r" (j), [k] "+r" (k), [l] "+r" (l));
}

void
f2 (void)
{
  __asm__ volatile (""
            : [a] "+r,m" (a), [b] "+r,m" (b), [c] "+r,m" (c), [d] "+r,m" (d),
              [e] "+r,m" (e), [f] "+r,m" (f), [g] "+r,m" (g), [h] "+r,m" (h),
              [i] "+r,m" (i), [j] "+r,m" (j), [k] "+r,m" (k), [l] "+r,m" (l));
}

void
f3 (void)
{
  __asm__ volatile (""
            : [a] "=r" (a), [b] "=r" (b), [c] "=r" (c), [d] "=r" (d),
              [e] "=r" (e), [f] "=r" (f), [g] "=r" (g), [h] "=r" (h),
              [i] "=r" (i), [j] "=r" (j), [k] "=r" (k), [l] "=r" (l)
            : "[a]" (a), "[b]" (b), "[c]" (c), "[d]" (d),
              "[e]" (e), "[f]" (f), "[g]" (g), "[h]" (h),
              "[i]" (i), "[j]" (j), "[k]" (k), "[l]" (l));
}

void
f4 (void)
{
  __asm__ volatile (""
            : [a] "=r,m" (a), [b] "=r,m" (b), [c] "=r,m" (c), [d] "=r,m" (d),
              [e] "=r,m" (e), [f] "=r,m" (f), [g] "=r,m" (g), [h] "=r,m" (h),
              [i] "=r,m" (i), [j] "=r,m" (j), [k] "=r,m" (k), [l] "=r,m" (l)
            : "[a],m" (a), "[b],m" (b), "[c],m" (c), "[d],m" (d),
              "[e],m" (e), "[f],m" (f), "[g],m" (g), "[h],m" (h),
              "[i],m" (i), "[j],m" (j), "[k],m" (k), "[l],m" (l));
}
