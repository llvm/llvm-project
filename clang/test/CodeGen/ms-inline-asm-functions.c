// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-pc-windows-msvc -fms-extensions -S -o - | FileCheck %s

// Yes, this is an assembly test from Clang, because we need to make it all the
// way through code generation to know if our call became a direct, pc-relative
// call or an indirect call through memory.

int k(int);
__declspec(dllimport) int kimport(int);
int (*kptr)(int);
int (*gptr(void))(int);

int foo(void) {
  // CHECK-LABEL: _foo:
  int (*r)(int) = gptr();

  // Simple case: direct call.
  __asm call k;
  // CHECK:     calll   _k

  // Marginally harder: indirect calls, via dllimport or function pointer.
  __asm call r;
  // CHECK:     calll   *({{.*}})
  __asm call kimport;
  // CHECK:     calll   *({{.*}})

  // Call through a global function pointer.
  __asm call kptr;
  // CHECK:     calll   *_kptr
}

int bar(void) {
  // CHECK-LABEL: _bar:
  __asm {
    jmp k
    ja k
    JAE k
    LOOP k
    loope k
    loopne k
  };
  // CHECK:      jmp     _k
  // CHECK-NEXT: ja      _k
  // CHECK-NEXT: jae     _k
  // CHECK-NEXT: loop    _k
  // CHECK-NEXT: loope   _k
  // CHECK-NEXT: loopne  _k
}

int baz(void) {
  // CHECK-LABEL: _baz:
  __asm mov eax, k;
  // CHECK: movl    _k, %eax
  __asm mov eax, kptr;
  // CHECK: movl    _kptr, %eax
}

// Test that this asm blob doesn't require more registers than available.  This
// has to be an LLVM code generation test.

void __declspec(naked) naked(void) {
  __asm pusha
  __asm call k
  __asm popa
  __asm ret
  // CHECK-LABEL: _naked:
  // CHECK: pushal
  // CHECK-NEXT: calll _k
  // CHECK-NEXT: popal
  // CHECK-NEXT: retl
}
