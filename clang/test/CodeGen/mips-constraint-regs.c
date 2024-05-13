// RUN: %clang -target mipsel-unknown-linux -S -o - -emit-llvm %s \
// RUN: | FileCheck %s

// This checks that the frontend will accept inline asm constraints
// c', 'l' and 'x'.

int main(void)
{
  // 'c': 16 bit address register for Mips16, GPR for all others
  // I am using 'c' to constrain both the target and one of the source
  // registers. We are looking for syntactical correctness.
  // CHECK: %{{[0-9]+}} = call i32 asm sideeffect "addi $0,$1,$2 \0A\09\09", "=c,c,I,~{$1}"(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}) [[NUW:#[0-9]+]], !srcloc !{{[0-9]+}}
  int __s, __v = 17;
  int __t;
  __asm__ __volatile__(
      "addi %0,%1,%2 \n\t\t"
      : "=c" (__t)
        : "c" (__s), "I" (__v));

  // 'l': lo register
  // We are making it clear that destination register is lo with the
  // use of the 'l' constraint ("=l").
  // CHECK:   %{{[0-9]+}} = call i32 asm sideeffect "mtlo $1 \0A\09\09", "=l,r,~{lo},~{$1}"(i32 %{{[0-9]+}}) [[NUW]], !srcloc !{{[0-9]+}}
  int i_temp = 44;
  int i_result;
  __asm__ __volatile__(
      "mtlo %1 \n\t\t"
      : "=l" (i_result)
        : "r" (i_temp)
          : "lo");

  // 'x': Combined lo/hi registers
  // We are specifying that destination registers are the hi/lo pair with the
  // use of the 'x' constraint ("=x").
  // CHECK:  %{{[0-9]+}} = call i64 asm sideeffect "mthi $1 \0A\09\09mtlo $2 \0A\09\09", "=x,r,r,~{$1}"(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}) [[NUW]], !srcloc !{{[0-9]+}}
  int i_hi = 3;
  int i_lo = 2;
  long long ll_result = 0;
  __asm__ __volatile__(
      "mthi %1 \n\t\t"
      "mtlo %2 \n\t\t"
      : "=x" (ll_result)
        : "r" (i_hi), "r" (i_lo)
          : );

  return 0;
}

// CHECK: attributes [[NUW]] = { nounwind }
