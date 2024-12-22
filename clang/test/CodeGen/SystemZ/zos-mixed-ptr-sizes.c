// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm -O2 < %s | FileCheck %s --check-prefix=X64

#define PSA_PTR 0x00
#define PSAAOLD 0x224

struct Foo {
  int * __ptr32 p32;
  int *p64;
  char *cp64;
};

void use_foo(struct Foo *f);

void ptr32_to_ptr(struct Foo *f, int * __ptr32 i) {
  // X64-LABEL: define void @ptr32_to_ptr(ptr noundef %f, ptr addrspace(1) noundef %i)
  // X64: %{{.+}} = addrspacecast ptr addrspace(1) %i to ptr
  f->p64= i;
  use_foo(f);
}

void ptr_to_ptr32(struct Foo *f, int *i) {
  // X64-LABEL: define void @ptr_to_ptr32(ptr noundef %f, ptr noundef %i)
  // X64: %{{.+}} = addrspacecast ptr %i to ptr addrspace(1)
  f->p32 = i;
  use_foo(f);
}

void ptr32_to_ptr32(struct Foo *f, int * __ptr32 i) {
  // X64-LABEL: define void @ptr32_to_ptr32(ptr noundef %f, ptr addrspace(1) noundef %i)
  // X64-NOT: addrspacecast
  f->p32 = i;
  use_foo(f);
}

void ptr_to_ptr32_explicit_cast(struct Foo *f, int *i) {
  // X64-LABEL: define void @ptr_to_ptr32_explicit_cast(ptr noundef %f, ptr noundef %i)
  // X64: %{{.+}} = addrspacecast ptr %i to ptr addrspace(1)
  f->p32 = (int * __ptr32)i;
  use_foo(f);
}

void test_indexing(struct Foo *f) {
  // X64-LABEL: define void @test_indexing(ptr noundef %f)
  // X64: addrspacecast ptr addrspace(1) {{%[0-9]}} to ptr
  f->cp64 = ((char * __ptr32 *)1028)[1];
  use_foo(f);
}

void test_indexing_2(struct Foo *f) {
  // X64-LABEL: define void @test_indexing_2(ptr noundef %f)
  // X64: getelementptr inbounds i8, ptr addrspace(1) {{%[0-9]}}, i32 16
  // X64: getelementptr inbounds i8, ptr {{%[0-9]}}, i64 24
  f->cp64 = ((char *** __ptr32 *)1028)[1][2][3];
  use_foo(f);
}

unsigned long* test_misc() {
  // X64-LABEL: define ptr @test_misc()
  // X64: %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %0, i32 88
  // X64-NEXT: %1 = load ptr, ptr addrspace(1) %arrayidx
  // X64-NEXT: %arrayidx1 = getelementptr inbounds i8, ptr %1, i64 8
  // X64-NEXT: %2 = load ptr, ptr %arrayidx1
  // X64-NEXT: %arrayidx2 = getelementptr inbounds i8, ptr %2, i64 904
  // X64-NEXT: %3 = load ptr, ptr %arrayidx2
  // X64-NEXT: %arrayidx3 = getelementptr inbounds i8, ptr %3, i64 1192
  unsigned long* x = (unsigned long*)((char***** __ptr32*)1208)[0][11][1][113][149];
  return x;
}

char* __ptr32* __ptr32 test_misc_2() {
  // X64-LABEL: define ptr addrspace(1) @test_misc_2()
  // X64: br i1 %cmp, label %if.then, label %if.end
  // X64: %1 = load ptr addrspace(1), ptr inttoptr (i64 16 to ptr)
  // X64-NEXT: %arrayidx = getelementptr inbounds i8, ptr addrspace(1) %1, i32 544
  // X64-NEXT: %2 = load ptr addrspace(1), ptr addrspace(1) %arrayidx
  // X64-NEXT: %arrayidx1 = getelementptr inbounds i8, ptr addrspace(1) %2, i32 24
  // X64-NEXT: %3 = load ptr addrspace(1), ptr addrspace(1) %arrayidx1
  // X64-NEXT: store ptr addrspace(1) %3, ptr @test_misc_2.res
  // X64: ret ptr addrspace(1)
  static char* __ptr32* __ptr32 res = 0;
  if (res == 0) {
    res = ((char* __ptr32* __ptr32* __ptr32* __ptr32*)0)[4][136][6];
  }
  return res;
}

unsigned short test_misc_3() {
  // X64-LABEL: define zeroext i16 @test_misc_3()
  // X64: %0 = load ptr addrspace(1), ptr inttoptr (i64 548 to ptr)
  // X64-NEXT: %1 = addrspacecast ptr addrspace(1) %0 to ptr
  // X64-NEXT: %arrayidx = getelementptr inbounds i8, ptr %1, i64 36
  // X64-NEXT: %2 = load i16, ptr %arrayidx, align 2
  // X64-NEXT: ret i16 %2
  unsigned short this_asid = ((unsigned short*)(*(char* __ptr32*)(0x224)))[18];
  return this_asid;
}

int test_misc_4() {
  // X64-LABEL: define signext range(i32 0, 2) i32 @test_misc_4()
  // X64: getelementptr inbounds i8, ptr addrspace(1) {{%[0-9]}}, i32 88
  // X64: getelementptr inbounds i8, ptr {{%[0-9]}}, i64 8
  // X64: getelementptr inbounds i8, ptr {{%[0-9]}}, i64 984
  // X64: getelementptr inbounds i8, ptr %3, i64 80
  // X64: icmp sgt i32 {{.*[0-9]}}, 67240703
  // X64: ret i32
  int a = (*(int*)(80 + ((char**** __ptr32*)1208)[0][11][1][123]) > 0x040202FF);
  return a;
}

void test_misc_5(struct Foo *f) {
  // X64-LABEL: define void @test_misc_5(ptr noundef %f)
  // X64: addrspacecast ptr addrspace(1) %0 to ptr
  f->cp64  = *(char* __ptr32 *)(PSA_PTR + PSAAOLD);
  use_foo(f);
}

int test_misc_6() {
  // X64-LABEL: define {{.*}} i32 @test_misc_6()
  // X64: ret i32 8
  int * __ptr32 ip32;
  int *ip64;
  ip64 = ip32;
  return sizeof(ip64);
}

int test_misc_7() {
  // X64-LABEL: define {{.*}} i32 @test_misc_7()
  // X64: ret i32 12
  int foo = 12;

  int *ip64;
  int * __ptr32 ip32;

  ip64 = &foo;
  ip32 = (int * __ptr32) ip64;

  return *ip32;
}

int test_misc_8() {
  // X64-LABEL: define {{.*}} i32 @test_misc_8()
  // X64: ret i32 97
  char foo = 'a';

  char *cp64;
  char * __ptr32 cp32;

  cp64 = &foo;
  cp32 = (char * __ptr32) cp64;

  return *cp32;
}

int test_misc_9() {
  // X64-LABEL: define {{.*}} i32 @test_misc_9()
  // X64: ret i32 15
  int foo = 15;

  int *ip64;
  int * __ptr32 ip32;

  ip32 = &foo;
  ip64 = (int *)ip32;

  return *ip64;
}

int test_misc_10() {
  // X64-LABEL: define {{.*}} i32 @test_misc_10()
  // X64: ret i32 97
  char foo = 'a';

  char *cp64;
  char * __ptr32 cp32;

  cp32 = &foo;
  cp64= (char *)cp32;

  return *cp64;
}

int test_function_ptr32_is_32bit() {
  // X64-LABEL: define {{.*}} i32 @test_function_ptr32_is_32bit()
  // X64: ret i32 4
  int (* __ptr32 a)(int a);
  return sizeof(a);
}

int get_processor_count() {
  // X64-LABEL: define signext range(i32 -128, 128) i32 @get_processor_count()
  // X64: load ptr addrspace(1), ptr inttoptr (i64 16 to ptr)
  // X64-NEXT: [[ARR_IDX1:%[a-z].*]] = getelementptr inbounds i8, ptr addrspace(1) %0, i32 660
  // X64: load ptr addrspace(1), ptr addrspace(1) [[ARR_IDX1]]
  // X64: load i8, ptr addrspace(1) {{%[a-z].*}}
  // X64: sext i8 {{%[0-9]}} to i32
  // X64-NEXT: ret i32
  return ((char * __ptr32 * __ptr32 *)0)[4][165][53];
}

int get_sizes_ptr32() {
  // X64-LABEL: define {{.*}} i32 @get_sizes_ptr32()
  // X64: ret i32 72
  char * __ptr32 a;
  signed char * __ptr32 b;
  unsigned char *__ptr32 c;
  int * __ptr32 d;
  signed int * __ptr32 e;
  unsigned int *__ptr32 f;
  short * __ptr32 g;
  signed short * __ptr32 h;
  unsigned short * __ptr32 i;
  long * __ptr32 j;
  signed * __ptr32 k;
  unsigned * __ptr32 l;
  long long * __ptr32 m;
  signed long long * __ptr32 n;
  unsigned long long * __ptr32 o;
  float * __ptr32 p;
  double * __ptr32 q;
  long double * __ptr32 r;

  int sum = 0;
  sum += sizeof(a);
  sum += sizeof(b);
  sum += sizeof(c);
  sum += sizeof(d);
  sum += sizeof(e);
  sum += sizeof(f);
  sum += sizeof(g);
  sum += sizeof(h);
  sum += sizeof(i);
  sum += sizeof(j);
  sum += sizeof(k);
  sum += sizeof(l);
  sum += sizeof(m);
  sum += sizeof(n);
  sum += sizeof(o);
  sum += sizeof(p);
  sum += sizeof(q);
  sum += sizeof(r);

  return sum;
}

int get_sizes_p64() {
  // X64-LABEL: define {{.*}} i32 @get_sizes_p64()
  // X64: ret i32 144
  char *a;
  signed char *b;
  unsigned char *c;
  int *d;
  signed int *e;
  unsigned int *f;
  short *g;
  signed short *h;
  unsigned short *i;
  long *j;
  signed *k;
  unsigned *l;
  long long *m;
  signed long long *n;
  unsigned long long *o;
  float *p;
  double *q;
  long double *r;

  int sum = 0;
  sum += sizeof(a);
  sum += sizeof(b);
  sum += sizeof(c);
  sum += sizeof(d);
  sum += sizeof(e);
  sum += sizeof(f);
  sum += sizeof(g);
  sum += sizeof(h);
  sum += sizeof(i);
  sum += sizeof(j);
  sum += sizeof(k);
  sum += sizeof(l);
  sum += sizeof(m);
  sum += sizeof(n);
  sum += sizeof(o);
  sum += sizeof(p);
  sum += sizeof(q);
  sum += sizeof(r);

  return sum;

}

int host_cpu() {
  char *__ptr32 CVT = *(char * __ptr32 *__ptr32) 16;
  unsigned short Id = *(unsigned short *)&CVT[-6];
  Id = ((((Id >> 12) & 0x0f) * 10 + ((Id >> 8) & 0x0f)) * 10 + ((Id >> 4) & 0x0f)) * 10 + (Id & 0x0f);
  int HaveVectorSupport = CVT[244] & 0x80;
  int z13 = (Id >= 2964 && HaveVectorSupport);
  return z13;
}
