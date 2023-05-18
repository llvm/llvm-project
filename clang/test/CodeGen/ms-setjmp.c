// RUN: %clang_cc1 -fms-extensions -DDECLARE_SETJMP -triple i686-windows-msvc   -emit-llvm %s -o - | FileCheck --check-prefix=I386 %s
// RUN: %clang_cc1 -fms-extensions -DDECLARE_SETJMP -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck --check-prefix=X64 %s
// RUN: %clang_cc1 -fms-extensions -DDECLARE_SETJMP -triple aarch64-windows-msvc -emit-llvm %s -o - | FileCheck --check-prefix=AARCH64 %s
// RUN: %clang_cc1 -fms-extensions -triple i686-windows-msvc -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck --check-prefix=I386 %s
// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck --check-prefix=X64 %s
// RUN: %clang_cc1 -fms-extensions -triple aarch64-windows-msvc -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck --check-prefix=AARCH64 %s
typedef char jmp_buf[1];

#ifdef DECLARE_SETJMP
int _setjmp(jmp_buf env);
int _setjmpex(jmp_buf env);
#endif

jmp_buf jb;

int test_setjmp(void) {
  return _setjmp(jb);
  // I386-LABEL: define dso_local i32 @test_setjmp
  // I386:       %[[call:.*]] = call i32 (ptr, i32, ...) @_setjmp3(ptr @jb, i32 0)
  // I386-NEXT:  ret i32 %[[call]]

  // X64-LABEL: define dso_local i32 @test_setjmp
  // X64:       %[[addr:.*]] = call ptr @llvm.frameaddress.p0(i32 0)
  // X64:       %[[call:.*]] = call i32 @_setjmp(ptr @jb, ptr %[[addr]])
  // X64-NEXT:  ret i32 %[[call]]

  // AARCH64-LABEL: define dso_local i32 @test_setjmp
  // AARCH64:       %[[addr:.*]] = call ptr @llvm.sponentry.p0()
  // AARCH64:       %[[call:.*]] = call i32 @_setjmpex(ptr @jb, ptr %[[addr]])
  // AARCH64-NEXT:  ret i32 %[[call]]
}

int test_setjmpex(void) {
  return _setjmpex(jb);
  // X64-LABEL: define dso_local i32 @test_setjmpex
  // X64:       %[[addr:.*]] = call ptr @llvm.frameaddress.p0(i32 0)
  // X64:       %[[call:.*]] = call i32 @_setjmpex(ptr @jb, ptr %[[addr]])
  // X64-NEXT:  ret i32 %[[call]]

  // AARCH64-LABEL: define dso_local i32 @test_setjmpex
  // AARCH64:       %[[addr:.*]] = call ptr @llvm.sponentry.p0()
  // AARCH64:       %[[call:.*]] = call i32 @_setjmpex(ptr @jb, ptr %[[addr]])
  // AARCH64-NEXT:  ret i32 %[[call]]
}
