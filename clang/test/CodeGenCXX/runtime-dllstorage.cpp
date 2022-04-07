// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-msvc -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-MS -check-prefix CHECK-MS-DYNAMIC
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-msvc -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-MS -check-prefix CHECK-MS-STATIC

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-DYNAMIC-IA -check-prefix CHECK-DYNAMIC-NODECL-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA -check-prefix CHECK-STATIC-NODECL-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -DIMPORT_DECLARATIONS -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-DYNAMIC-IA -check-prefix CHECK-DYNAMIC-IMPORT-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -DIMPORT_DECLARATIONS -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA -check-prefix CHECK-STATIC-IMPORT-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -DEXPORT_DECLARATIONS -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-DYNAMIC-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -DEXPORT_DECLARATIONS -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -DDECL -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-DYNAMIC-IA -check-prefix CHECK-DYNAMIC-DECL-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -DDECL -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA -check-prefix CHECK-STATIC-DECL-IA
// %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -fno-use-cxa-atexit -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-IA -check-prefix CHECK-DYNAMIC-IA -check-prefix CHECK-DYNAMIC-IA-ATEXIT
// %clang_cc1 -no-opaque-pointers -triple i686-windows-itanium -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -fno-use-cxa-atexit -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA -check-prefix CHECK-STATIC-IA-ATEXIT

// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-gnu -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-gnu -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-cygnus -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA
// RUN: %clang_cc1 -no-opaque-pointers -triple i686-windows-cygnus -std=c++11 -fdeclspec -fms-compatibility -fexceptions -fcxx-exceptions -flto-visibility-public-std -emit-llvm -o - %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix CHECK-IA -check-prefix CHECK-STATIC-IA

#if defined(IMPORT_DECLARATIONS)
namespace __cxxabiv1 {
extern "C" {
__declspec(dllimport) void __cxa_guard_acquire(unsigned long long *);
__declspec(dllimport) unsigned char *__cxa_allocate_exception(unsigned long);
}
extern "C" __declspec(dllimport) void __cxa_guard_release(unsigned long long *);
}
namespace std {
__declspec(dllimport) __declspec(noreturn) void terminate();
}
#elif defined(EXPORT_DECLARATIONS)
namespace __cxxabiv1 {
extern "C" {
__declspec(dllexport) void __cxa_guard_acquire(unsigned long long *);
__declspec(dllexport) unsigned char *__cxa_allocate_exception(unsigned long);
}
extern "C" void __declspec(dllexport) __cxa_guard_release(unsigned long long *);
}
namespace std {
__declspec(dllexport) __declspec(noreturn) void terminate();
}
#elif defined(DECL)
namespace __cxxabiv1 {
extern "C" unsigned char *__cxa_allocate_exception(unsigned long);
}
namespace std {
__declspec(noreturn) void terminate();
}
#else
namespace std {
__declspec(noreturn) void terminate();
}
#endif

struct s {
  s() = default;
  s(unsigned char) { throw 0; }
  int m() const;
};

struct t {
  ~t();
  int m() const;
};

struct u {
  ~u();
};

s o;
thread_local t t;
u v;
__declspec(thread) s q;

void __declspec(noinline) f() {
  throw 0;
}

void g();
void __declspec(noinline) h() {
  try {
    g();
  } catch (const int &) {
    return;
  } catch (...) {
    throw;
  }
}

void i() {
  s r(static_cast<unsigned char>('\t'));
}

int j() {
  static thread_local struct t v;
  static struct t *w = new struct t;
  return w->m() ? v.m() : w->m();
}

void k() noexcept {
  g();
}

void l() {
  std::terminate();
}

// CHECK-MS-DAG: @_Init_thread_epoch = external thread_local global i32
// CHECK-MS-DAG: declare dso_local i32 @__tlregdtor(void ()*)
// CHECK-MS-DAG: declare dso_local i32 @atexit(void ()*)
// CHECK-MS-DYNAMIC-DAG: declare {{.*}} void @_CxxThrowException
// CHECK-MS-STATIC-DAG: declare {{.*}} void @_CxxThrowException
// CHECK-MS-DAG: declare dso_local noundef nonnull i8* @"??2@YAPAXI@Z"
// CHECK-MS-DAG: declare dso_local void @_Init_thread_header(i32*)
// CHECK-MS-DAG: declare dso_local void @_Init_thread_footer(i32*)

// CHECK-IA-DAG: @_ZTH1t = dso_local alias void (), void ()* @__tls_init
// CHECK-IA-DAG: declare dso_local i32 @__gxx_personality_v0(...)
// CHECK-IA-DAG: define linkonce_odr hidden void @__clang_call_terminate(i8* %0)

// CHECK-DYNAMIC-IA-DAG: declare dllimport i32 @__cxa_thread_atexit(void (i8*)*, i8*, i8*)
// CHECK-DYNAMIC-IA-DAG: declare dllimport i32 @__cxa_atexit(void (i8*)*, i8*, i8*)
// CHECK-DYNAMIC-IA-DECL-DAG: declare i8* @__cxa_allocate_exception(i32 noundef)
// CHECK-DYNAMIC-IA-NODECL-DAG: declare dllimport i8* @__cxa_allocate_exception(i32 noundef)
// CHECK-DYNAMIC-IA-IMPORT-DAG: declare dllimport i8* @__cxa_allocate_exception(i32 noundef)
// CHECK-DYNAMIC-IA-EXPORT-DAG: declare dllimport i8* @__cxa_allocate_exception(i32 noundef)
// CHECK-DYNAMIC-IA-DAG: declare dllimport void @__cxa_throw(i8*, i8*, i8*)
// CHECK-DYNAMIC-DECL-IA-DAG: declare dllimport i32 @__cxa_guard_acquire(i64*)
// CHECK-DYNAMIC-NODECL-IA-DAG: declare dllimport i32 @__cxa_guard_acquire(i64*)
// CHECK-DYNAMIC-IMPORT-IA-DAG: declare dllimport i32 @__cxa_guard_acquire(i64*)
// CHECK-DYNAMIC-EXPORT-IA-DAG: declare dllexport i32 @__cxa_guard_acquire(i64*)
// CHECK-IA-DAG: declare dso_local noundef nonnull i8* @_Znwj(i32 noundef)
// CHECK-DYNAMIC-DECL-IA-DAG: declare dllimport void @__cxa_guard_release(i64*)
// CHECK-DYNAMIC-NODECL-IA-DAG: declare dllimport void @__cxa_guard_release(i64*)
// CHECK-DYNAMIC-IMPORT-IA-DAG: declare dllimport void @__cxa_guard_release(i64*)
// CHECK-DYNAMIC-EXPORT-IA-DAG: declare dllimport void @__cxa_guard_release(i64*)
// CHECK-DYANMIC-IA-DAG: declare dllimport void @_ZSt9terminatev()
// CHECK-DYNAMIC-NODECL-IA-DAG: declare dso_local void @_ZSt9terminatev()
// CHECK-DYNAMIC-IMPORT-IA-DAG: declare dllimport void @_ZSt9terminatev()
// CHECK-DYNAMIC-EXPORT-IA-DAG: declare dso_local dllexport void @_ZSt9terminatev()

// CHECK-STATIC-IA-DAG: declare dso_local i32 @__cxa_thread_atexit(void (i8*)*, i8*, i8*)
// CHECK-STATIC-IA-DAG: declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*)
// CHECK-STATIC-IA-DAG: declare dso_local i8* @__cxa_allocate_exception(i32)
// CHECK-STATIC-IA-DAG: declare dso_local void @__cxa_throw(i8*, i8*, i8*)
// CHECK-STATIC-DECL-IA-DAG: declare dso_local i32 @__cxa_guard_acquire(i64*)
// CHECK-STATIC-NODECL-IA-DAG: declare dso_local i32 @__cxa_guard_acquire(i64*)
// CHECK-STATIC-IMPORT-IA-DAG: declare dso_local i32 @__cxa_guard_acquire(i64*)
// CHECK-STATIC-EXPORT-IA-DAG: declare dso_local i32 @__cxa_guard_acquire(i64*)
// CHECK-IA-DAG: declare dso_local noundef nonnull i8* @_Znwj(i32 noundef)
// CHECK-STATIC-DECL-IA-DAG: declare dso_local void @__cxa_guard_release(i64*)
// CHECK-STATIC-NODECL-IA-DAG: declare dso_local void @__cxa_guard_release(i64*)
// CHECK-STATIC-IMPORT-IA-DAG: declare dso_local void @__cxa_guard_release(i64*)
// CHECK-STATIC-EXPORT-IA-DAG: declare dso_local void @__cxa_guard_release(i64*)
// CHECK-STATIC-IA-DAG: declare dso_local void @_ZSt9terminatev()
// CHECK-STATIC-NODECL-IA-DAG: declare dso_local void @_ZSt9terminatev()
// CHECK-STATIC-IMPORT-IA-DAG: declare dso_local void @_ZSt9terminatev()
// CHECK-STATIC-EXPORT-IA-DAG: declare dso_local dllexport void @_ZSt9terminatev()
