// RUN: %clang_cc1 -O0 -emit-llvm -Wno-error=return-type -ftrapv -ftrap-function=mytrap %s -o - | FileCheck %s -check-prefix=TRAPFUNC
// RUN: %clang_cc1 -O0 -emit-llvm -Wno-error=return-type -ftrapv %s -o - | FileCheck %s -check-prefix=NOOPTION

// TRAPFUNC-LABEL: define {{(dso_local )?}}void @{{_Z12test_builtinv|\"\?test_builtin@@YAXXZ\"}}
// TRAPFUNC: call void @llvm.trap() [[ATTR0:#[0-9]+]]

// NOOPTION-LABEL: define {{(dso_local )?}}void @{{_Z12test_builtinv|\"\?test_builtin@@YAXXZ\"}}
// NOOPTION: call void @llvm.trap(){{$}}

void test_builtin(void) {
  __builtin_trap();
}

// TRAPFUNC-LABEL: define {{.*}}i32 @{{_Z13test_noreturnv|\"\?test_noreturn@@YAHXZ\"}}
// TRAPFUNC: call void @llvm.trap() [[ATTR0]]

// NOOPTION-LABEL: define {{.*}}i32 @{{_Z13test_noreturnv|\"\?test_noreturn@@YAHXZ\"}}
// NOOPTION: call void @llvm.trap(){{$}}

int test_noreturn(void) {
}

// TRAPFUNC-LABEL: define {{.*}}i32 @{{_Z17test_add_overflowii|\"\?test_add_overflow@@YAHHH@Z\"}}
// TRAPFUNC: call void @llvm.ubsantrap({{.*}}) [[ATTR1:#[0-9]+]]

// NOOPTION-LABEL: define {{.*}}i32 @{{_Z17test_add_overflowii|\"\?test_add_overflow@@YAHHH@Z\"}}
// NOOPTION: call void @llvm.ubsantrap({{.*}}) [[ATTR2:#[0-9]+]]

int test_add_overflow(int a, int b) {
  return a + b;
}

// TRAPFUNC: attributes [[ATTR0]] = { {{.*}}"trap-func-name"="mytrap" }
// TRAPFUNC: attributes [[ATTR1]] = { {{.*}}"trap-func-name"="mytrap" }

// NOOPTION-NOT: attributes [[ATTR2]] = { {{.*}}"trap-func-name"="mytrap" }
