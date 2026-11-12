// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify -triple i686-pc-win32 %s
// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify -triple x86_64-pc-win32 %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify -triple i686-pc-win32 %s
// RUN: %clang_cc1 -x c++ -DEXTERN_C='extern "C"' -fsyntax-only -fms-extensions -verify -triple i686-pc-win32 %s

#ifndef EXTERN_C
#define EXTERN_C
#if defined(__cplusplus)
#define EXPECT_NODIAG
// expected-no-diagnostics
#endif
#endif

#ifndef EXPECT_NODIAG
// expected-note-re@+2 1+ {{forward declaration of '{{(struct )?}}Foo'}}
#endif
struct Foo;

EXTERN_C void __stdcall fwd_std(struct Foo p);
#if !defined(EXPECT_NODIAG) && defined(_M_IX86)
// expected-error@+2 {{parameter 'p' must have a complete type to use function 'fwd_std' with the stdcall calling convention}}
#endif
void (__stdcall *fp_fwd_std)(struct Foo) = &fwd_std;

EXTERN_C void __fastcall fwd_fast(struct Foo p);
#if !defined(EXPECT_NODIAG) && defined(_M_IX86)
// expected-error@+2 {{parameter 'p' must have a complete type to use function 'fwd_fast' with the fastcall calling convention}}
#endif
void (__fastcall *fp_fwd_fast)(struct Foo) = &fwd_fast;

EXTERN_C void __vectorcall fwd_vector(struct Foo p);
#if !defined(EXPECT_NODIAG)
// expected-error@+2 {{parameter 'p' must have a complete type to use function 'fwd_vector' with the vectorcall calling convention}}
#endif
void (__vectorcall *fp_fwd_vector)(struct Foo) = &fwd_vector;

#if defined(__cplusplus)
template <typename T> struct TemplateWrapper {
#ifndef EXPECT_NODIAG
  // expected-error@+2 {{field has incomplete type 'Foo'}}
#endif
  T field;
};

EXTERN_C void __vectorcall tpl_ok(TemplateWrapper<int> p);
void(__vectorcall *fp_tpl_ok)(TemplateWrapper<int>) = &tpl_ok;

EXTERN_C void __vectorcall tpl_fast(TemplateWrapper<Foo> p);
#ifndef EXPECT_NODIAG
// expected-note@+2 {{requested here}}
#endif
void(__vectorcall *fp_tpl_fast)(TemplateWrapper<Foo>) = &tpl_fast;
#endif
