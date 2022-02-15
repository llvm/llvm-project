// RUN: %clang_cc1 -O2 -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu | FileCheck %s

int test_cca(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_cca
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@cca},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@cca"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccae(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccae
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccae},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccae"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccb(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccb
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccb},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccb"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccbe(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccbe
  //CHECK: tail call i32 asm "cmp $2,$1", "={@ccbe},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccbe"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccc(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccc
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccc},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccc"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_cce(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_cce
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@cce},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@cce"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccz(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccz
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccz},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccz"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccg(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccg
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccg},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccg"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccge(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccge
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccge},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccge"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccl(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccl
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccl},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccl"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccle(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccle
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccle},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccle"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccna(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccna
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccna},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccna"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnae(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnae
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnae},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnae"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnb(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnb
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnb},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnb"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnbe(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnbe
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnbe},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnbe"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnc(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnc
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnc},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnc"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccne(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccne
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccne},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccne"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnz(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnz
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnz},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnz"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccng(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccng
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccng},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccng"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnge(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnge
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnge},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnge"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnl(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnl
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnl},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnl"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnle(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnle
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnle},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnle"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccno(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccno
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccno},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccno"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccnp(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccnp
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccnp},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccnp"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccns(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccns
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccns},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccns"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_cco(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_cco
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@cco},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@cco"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccp(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccp
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccp},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccp"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

int test_ccs(long nr, volatile long *addr) {
  //CHECK-LABEL: @test_ccs
  //CHECK: = tail call i32 asm "cmp $2,$1", "={@ccs},=*m,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i64* elementtype(i64) %addr, i64 %nr)
  int x;
  asm("cmp %2,%1"
      : "=@ccs"(x), "=m"(*(volatile long *)(addr))
      : "r"(nr)
      : "cc");
  if (x)
    return 0;
  return 1;
}

_Bool check_no_clobber_conflicts(void) {
  //CHECK-LABEL: @check_no_clobber_conflicts
  //CHECK:  = tail call i8 asm "", "={@cce},~{cx},~{dirflag},~{fpsr},~{flags}"()
  _Bool b;
  asm(""
      : "=@cce"(b)
      :
      : "cx");
  return b;
}
