// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -triple i686-windows-msvc   -emit-llvm -o - %s -fdefault-calling-conv=vectorcall | FileCheck %s --check-prefix=X86-VEC
// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s -fdefault-calling-conv=vectorcall | FileCheck %s --check-prefix=X64-VEC

void foo_default(const char *lpString1, const char *lpString2);
void __stdcall foo_std(const char *lpString1, const char *lpString2);
void __fastcall foo_fast(const char *lpString1, const char *lpString2);
void __vectorcall foo_vector(const char *lpString1, const char *lpString2);

void __cdecl bar(void) {
  foo_default(0, 0);
  foo_std(0, 0);
  foo_fast(0, 0);
  foo_vector(0, 0);
}

// X86-LABEL: define dso_local void @bar()
// X86:   call void @foo_default(ptr noundef null, ptr noundef null)
// X86:   call x86_stdcallcc void @"\01_foo_std@8"(ptr noundef null, ptr noundef null)
// X86:   call x86_fastcallcc void @"\01@foo_fast@8"(ptr inreg noundef null, ptr inreg noundef null)
// X86:   call x86_vectorcallcc void @"\01foo_vector@@8"(ptr inreg noundef null, ptr inreg noundef null)
// X86:   ret void

// X64-LABEL: define dso_local void @bar()
// X64:   call void @foo_default(ptr noundef null, ptr noundef null)
// X64:   call void @foo_std(ptr noundef null, ptr noundef null)
// X64:   call void @foo_fast(ptr noundef null, ptr noundef null)
// X64:   call x86_vectorcallcc void @"\01foo_vector@@16"(ptr noundef null, ptr noundef null)
// X64:   ret void

// X86-VEC-LABEL: define dso_local void @bar()
// X86-VEC:   call x86_vectorcallcc void @"\01foo_default@@8"(ptr inreg noundef null, ptr inreg noundef null)
// X86-VEC:   call x86_stdcallcc void @"\01_foo_std@8"(ptr noundef null, ptr noundef null)
// X86-VEC:   call x86_fastcallcc void @"\01@foo_fast@8"(ptr inreg noundef null, ptr inreg noundef null)
// X86-VEC:   call x86_vectorcallcc void @"\01foo_vector@@8"(ptr inreg noundef null, ptr inreg noundef null)
// X86-VEC:   ret void

// X64-VEC-LABEL: define dso_local void @bar()
// X64-VEC:   call x86_vectorcallcc void @"\01foo_default@@16"(ptr noundef null, ptr noundef null)
// X64-VEC:   call void @foo_std(ptr noundef null, ptr noundef null)
// X64-VEC:   call void @foo_fast(ptr noundef null, ptr noundef null)
// X64-VEC:   call x86_vectorcallcc void @"\01foo_vector@@16"(ptr noundef null, ptr noundef null)
// X64-VEC:   ret void

