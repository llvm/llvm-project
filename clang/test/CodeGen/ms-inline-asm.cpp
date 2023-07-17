// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -x c++ %s -triple i386-apple-darwin10 -fasm-blocks -emit-llvm -o - -std=c++11 | FileCheck %s

// rdar://13645930

struct Foo {
  static int *ptr;
  static int a, b;
  int arr[4];
  struct Bar {
    static int *ptr;
    char arr[2];
  };
};

void t1() {
// CHECK-LABEL: define{{.*}} void @_Z2t1v()
  Foo::ptr = (int *)0xDEADBEEF;
  Foo::Bar::ptr = (int *)0xDEADBEEF;
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov eax, $0
// CHECK-SAME: mov eax, $1
// CHECK-SAME: mov eax, $2
// CHECK-SAME: mov eax, dword ptr $3
// CHECK-SAME: mov eax, dword ptr $4
// CHECK-SAME: "*m,*m,*m,*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(ptr) @_ZN3Foo3ptrE, ptr elementtype(ptr) @_ZN3Foo3Bar3ptrE, ptr elementtype(ptr) @_ZN3Foo3ptrE, ptr elementtype(ptr) @_ZN3Foo3ptrE, ptr elementtype(ptr) @_ZN3Foo3ptrE)
  __asm mov eax, Foo ::ptr
  __asm mov eax, Foo :: Bar :: ptr
  __asm mov eax, [Foo:: ptr]
  __asm mov eax, dword ptr [Foo :: ptr]
  __asm mov eax, dword ptr [Foo :: ptr]
}

int gvar = 10;
void t2() {
  int lvar = 10;
  __asm mov eax, offset Foo::ptr
  __asm mov eax, offset Foo::Bar::ptr
// CHECK-LABEL: define{{.*}} void @_Z2t2v()
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov eax, $0
// CHECK-SAME: mov eax, $1
// CHECK-SAME: "i,i,~{eax},~{dirflag},~{fpsr},~{flags}"(ptr @_ZN3Foo3ptrE, ptr @_ZN3Foo3Bar3ptrE)
}

// CHECK-LABEL: define{{.*}} void @_Z2t3v()
void t3() {
  __asm mov eax, LENGTH Foo::ptr
  __asm mov eax, LENGTH Foo::Bar::ptr
  __asm mov eax, LENGTH Foo::arr
  __asm mov eax, LENGTH Foo::Bar::arr

  __asm mov eax, TYPE Foo::ptr
  __asm mov eax, TYPE Foo::Bar::ptr
  __asm mov eax, TYPE Foo::arr
  __asm mov eax, TYPE Foo::Bar::arr

  __asm mov eax, SIZE Foo::ptr
  __asm mov eax, SIZE Foo::Bar::ptr
  __asm mov eax, SIZE Foo::arr
  __asm mov eax, SIZE Foo::Bar::arr
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov eax, $$1
// CHECK-SAME: mov eax, $$1
// CHECK-SAME: mov eax, $$4
// CHECK-SAME: mov eax, $$2
// CHECK-SAME: mov eax, $$4
// CHECK-SAME: mov eax, $$4
// CHECK-SAME: mov eax, $$4
// CHECK-SAME: mov eax, $$1
// CHECK-SAME: mov eax, $$4
// CHECK-SAME: mov eax, $$4
// CHECK-SAME: mov eax, $$16
// CHECK-SAME: mov eax, $$2
// CHECK-SAME: "~{eax},~{dirflag},~{fpsr},~{flags}"()

}

struct T4 {
  int x;
  static int y;
  void test();
};

// CHECK-LABEL: define{{.*}} void @_ZN2T44testEv(
void T4::test() {
// CHECK: [[T0:%.*]] = alloca ptr,
// CHECK: [[THIS:%.*]] = load ptr, ptr [[T0]]
// CHECK: [[X:%.*]] = getelementptr inbounds [[T4:%.*]], ptr [[THIS]], i32 0, i32 0
  __asm mov eax, x;
  __asm mov y, eax;
// CHECK: call void asm sideeffect inteldialect
// CHECK-SAME: mov eax, $1
// CHECK-SAME: mov $0, eax
// CHECK-SAME: "=*m,*m,~{eax},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @_ZN2T41yE, ptr elementtype(i32) {{.*}})
}

template <class T> struct T5 {
  template <class U> static T create(U);
  void run();
};
// CHECK-LABEL: define{{.*}} void @_Z5test5v()
void test5() {
  // CHECK: [[X:%.*]] = alloca i32
  // CHECK: [[Y:%.*]] = alloca i32
  int x, y;
  __asm push y
  __asm call T5<int>::create<float>
  __asm mov x, eax
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: push $0
  // CHECK-SAME: call ${2:P}
  // CHECK-SAME: mov $1, eax
  // CHECK-SAME: "=*m,=*m,*m,~{esp},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %y, ptr elementtype(i32) %x, ptr elementtype(i32 (float)) @_ZN2T5IiE6createIfEEiT_)
}

// Just verify this doesn't emit an error.
void test6() {
  __asm {
   a:
   jmp a
  }
}

void t7_struct() {
  struct A {
    int a;
    int b;
  };
  __asm mov eax, [eax].A.b
  // CHECK-LABEL: define{{.*}} void @_Z9t7_structv
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: mov eax, [eax + $$4]
  // CHECK-SAME: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t7_typedef() {
  typedef struct {
    int a;
    int b;
  } A;
  __asm mov eax, [eax].A.b
  // CHECK-LABEL: define{{.*}} void @_Z10t7_typedefv
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: mov eax, [eax + $$4]
  // CHECK-SAME: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t7_using() {
  using A = struct {
    int a;
    int b;
  };
  __asm mov eax, [eax].A.b
  // CHECK-LABEL: define{{.*}} void @_Z8t7_usingv
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: mov eax, [eax + $$4]
  // CHECK-SAME: "~{eax},~{dirflag},~{fpsr},~{flags}"()
}

void t8() {
  __asm some_label:
  // CHECK-LABEL: define{{.*}} void @_Z2t8v()
  // CHECK: call void asm sideeffect inteldialect
  // CHECK-SAME: L__MSASMLABEL_.${:uid}__some_label:
  // CHECK-SAME: "~{dirflag},~{fpsr},~{flags}"()
  struct A {
    static void g() {
      __asm jmp some_label ; This should jump forwards
      __asm some_label:
      __asm nop
      // CHECK-LABEL: define internal void @_ZZ2t8vEN1A1gEv()
      // CHECK: call void asm sideeffect inteldialect
      // CHECK-SAME: jmp L__MSASMLABEL_.${:uid}__some_label
      // CHECK-SAME: L__MSASMLABEL_.${:uid}__some_label:
      // CHECK-SAME: nop
      // CHECK-SAME: "~{dirflag},~{fpsr},~{flags}"()
    }
  };
  A::g();
}

void t9() {
  // CHECK-LABEL: define{{.*}} void @_Z2t9v()
  struct A {
    int a;
    int b;
    void g() {
      __asm mov eax, dword ptr [eax]this.b
      // CHECK: call void asm sideeffect inteldialect
      // CHECK-SAME: mov eax, dword ptr [eax + $$4]
      // CHECK-SAME: "~{eax},~{dirflag},~{fpsr},~{flags}"()
    }
  };
  A AA;
  AA.g();
}

