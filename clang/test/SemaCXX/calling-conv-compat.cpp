// RUN: %clang_cc1 -fsyntax-only -std=c++11 -fms-extensions -verify -triple i686-pc-win32 %s

// Pointers to free functions
void            free_func_default();
void __cdecl    free_func_cdecl();
void __stdcall  free_func_stdcall();
void __fastcall free_func_fastcall();

typedef void (           *fptr_default)();
typedef void (__cdecl    *fptr_cdecl)();
typedef void (__stdcall  *fptr_stdcall)();
typedef void (__fastcall *fptr_fastcall)();

// expected-note@+4 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'fptr_default' (aka 'void (*)()') for 1st argument}}
// expected-note@+3 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'fptr_default' (aka 'void (*)()') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (*)() __attribute__((stdcall))' to 'fptr_default' (aka 'void (*)()') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (*)() __attribute__((fastcall))' to 'fptr_default' (aka 'void (*)()') for 1st argument}}
void cb_fptr_default(fptr_default ptr);
// expected-note@+4 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'fptr_cdecl' (aka 'void (*)()') for 1st argument}}
// expected-note@+3 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'fptr_cdecl' (aka 'void (*)()') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (*)() __attribute__((stdcall))' to 'fptr_cdecl' (aka 'void (*)()') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (*)() __attribute__((fastcall))' to 'fptr_cdecl' (aka 'void (*)()') for 1st argument}}
void cb_fptr_cdecl(fptr_cdecl ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void ()' to 'fptr_stdcall' (aka 'void (*)() __attribute__((stdcall))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((cdecl))' to 'fptr_stdcall' (aka 'void (*)() __attribute__((stdcall))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'fptr_stdcall' (aka 'void (*)() __attribute__((stdcall))') for 1st argument}}
void cb_fptr_stdcall(fptr_stdcall ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void ()' to 'fptr_fastcall' (aka 'void (*)() __attribute__((fastcall))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((cdecl))' to 'fptr_fastcall' (aka 'void (*)() __attribute__((fastcall))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'fptr_fastcall' (aka 'void (*)() __attribute__((fastcall))') for 1st argument}}
void cb_fptr_fastcall(fptr_fastcall ptr);
// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'const fptr_default' (aka 'void (*const)()') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'const fptr_default' (aka 'void (*const)()') for 1st argument}}
void cb_fptr_const_default(const fptr_default ptr);

void call_free_func() {
  cb_fptr_default(free_func_default);
  cb_fptr_default(free_func_cdecl);
  cb_fptr_default(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fptr_default'}}
  cb_fptr_default(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fptr_default'}}
  cb_fptr_default(&free_func_default);
  cb_fptr_default(&free_func_cdecl);
  cb_fptr_default(&free_func_stdcall); // expected-error {{no matching function for call to 'cb_fptr_default'}}
  cb_fptr_default(&free_func_fastcall); // expected-error {{no matching function for call to 'cb_fptr_default'}}

  cb_fptr_cdecl(free_func_default);
  cb_fptr_cdecl(free_func_cdecl);
  cb_fptr_cdecl(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fptr_cdecl'}}
  cb_fptr_cdecl(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fptr_cdecl'}}
  cb_fptr_cdecl(&free_func_default);
  cb_fptr_cdecl(&free_func_cdecl);
  cb_fptr_cdecl(&free_func_stdcall); // expected-error {{no matching function for call to 'cb_fptr_cdecl'}}
  cb_fptr_cdecl(&free_func_fastcall); // expected-error {{no matching function for call to 'cb_fptr_cdecl'}}

  cb_fptr_stdcall(free_func_default); // expected-error {{no matching function for call to 'cb_fptr_stdcall'}}
  cb_fptr_stdcall(free_func_cdecl); // expected-error {{no matching function for call to 'cb_fptr_stdcall'}}
  cb_fptr_stdcall(free_func_stdcall);
  cb_fptr_stdcall(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fptr_stdcall'}}

  cb_fptr_fastcall(free_func_default); // expected-error {{no matching function for call to 'cb_fptr_fastcall'}}
  cb_fptr_fastcall(free_func_cdecl); // expected-error {{no matching function for call to 'cb_fptr_fastcall'}}
  cb_fptr_fastcall(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fptr_fastcall'}}
  cb_fptr_fastcall(free_func_fastcall);

  cb_fptr_const_default(free_func_default);
  cb_fptr_const_default(free_func_cdecl);
  cb_fptr_const_default(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fptr_const_default'}}
  cb_fptr_const_default(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fptr_const_default'}}

}

// Pointers to variadic functions
// variadic function can't declared stdcall or fastcall
void         free_func_variadic_default(int, ...);
void __cdecl free_func_variadic_cdecl(int, ...);

typedef void (        *fptr_variadic_default)(int, ...);
typedef void (__cdecl *fptr_variadic_cdecl)(int, ...);

void cb_fptr_variadic_default(fptr_variadic_default ptr);
void cb_fptr_variadic_cdecl(fptr_variadic_cdecl ptr);

void call_free_variadic_func() {
  cb_fptr_variadic_default(free_func_variadic_default);
  cb_fptr_variadic_default(free_func_variadic_cdecl);
  cb_fptr_variadic_default(&free_func_variadic_default);
  cb_fptr_variadic_default(&free_func_variadic_cdecl);

  cb_fptr_variadic_cdecl(free_func_variadic_default);
  cb_fptr_variadic_cdecl(free_func_variadic_cdecl);
  cb_fptr_variadic_cdecl(&free_func_variadic_default);
  cb_fptr_variadic_cdecl(&free_func_variadic_cdecl);
}

// References to functions
typedef void (           &fref_default)();
typedef void (__cdecl    &fref_cdecl)();
typedef void (__stdcall  &fref_stdcall)();
typedef void (__fastcall &fref_fastcall)();

// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'fref_default' (aka 'void (&)()') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'fref_default' (aka 'void (&)()') for 1st argument}}
void cb_fref_default(fref_default ptr);
// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'fref_cdecl' (aka 'void (&)()') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'fref_cdecl' (aka 'void (&)()') for 1st argument}}
void cb_fref_cdecl(fref_cdecl ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void ()' to 'fref_stdcall' (aka 'void (&)() __attribute__((stdcall))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((cdecl))' to 'fref_stdcall' (aka 'void (&)() __attribute__((stdcall))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((fastcall))' to 'fref_stdcall' (aka 'void (&)() __attribute__((stdcall))') for 1st argument}}
void cb_fref_stdcall(fref_stdcall ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void ()' to 'fref_fastcall' (aka 'void (&)() __attribute__((fastcall))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void () __attribute__((cdecl))' to 'fref_fastcall' (aka 'void (&)() __attribute__((fastcall))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void () __attribute__((stdcall))' to 'fref_fastcall' (aka 'void (&)() __attribute__((fastcall))') for 1st argument}}
void cb_fref_fastcall(fref_fastcall ptr);

void call_free_func_ref() {
  cb_fref_default(free_func_default);
  cb_fref_default(free_func_cdecl);
  cb_fref_default(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fref_default'}}
  cb_fref_default(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fref_default'}}

  cb_fref_cdecl(free_func_default);
  cb_fref_cdecl(free_func_cdecl);
  cb_fref_cdecl(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fref_cdecl'}}
  cb_fref_cdecl(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fref_cdecl'}}

  cb_fref_stdcall(free_func_default); // expected-error {{no matching function for call to 'cb_fref_stdcall'}}
  cb_fref_stdcall(free_func_cdecl); // expected-error {{no matching function for call to 'cb_fref_stdcall'}}
  cb_fref_stdcall(free_func_stdcall);
  cb_fref_stdcall(free_func_fastcall); // expected-error {{no matching function for call to 'cb_fref_stdcall'}}

  cb_fref_fastcall(free_func_default); // expected-error {{no matching function for call to 'cb_fref_fastcall'}}
  cb_fref_fastcall(free_func_cdecl); // expected-error {{no matching function for call to 'cb_fref_fastcall'}}
  cb_fref_fastcall(free_func_stdcall); // expected-error {{no matching function for call to 'cb_fref_fastcall'}}
  cb_fref_fastcall(free_func_fastcall);
}

// References to variadic functions
// variadic function can't declared stdcall or fastcall
typedef void (        &fref_variadic_default)(int, ...);
typedef void (__cdecl &fref_variadic_cdecl)(int, ...);

void cb_fref_variadic_default(fptr_variadic_default ptr);
void cb_fref_variadic_cdecl(fptr_variadic_cdecl ptr);

void call_free_variadic_func_ref() {
  cb_fref_variadic_default(free_func_variadic_default);
  cb_fref_variadic_default(free_func_variadic_cdecl);

  cb_fref_variadic_cdecl(free_func_variadic_default);
  cb_fref_variadic_cdecl(free_func_variadic_cdecl);
}

// Pointers to members
namespace NonVariadic {

struct A {
  void            member_default();
  void __cdecl    member_cdecl();
  void __thiscall member_thiscall();
};

struct B : public A {
};

struct C {
  void            member_default();
  void __cdecl    member_cdecl();
  void __thiscall member_thiscall();
};

typedef void (           A::*memb_a_default)();
typedef void (__cdecl    A::*memb_a_cdecl)();
typedef void (__thiscall A::*memb_a_thiscall)();
typedef void (           B::*memb_b_default)();
typedef void (__cdecl    B::*memb_b_cdecl)();
typedef void (__thiscall B::*memb_b_thiscall)();
typedef void (           C::*memb_c_default)();
typedef void (__cdecl    C::*memb_c_cdecl)();
typedef void (__thiscall C::*memb_c_thiscall)();

// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_a_default' (aka 'void (NonVariadic::A::*)() __attribute__((thiscall))') for 1st argument}}
void cb_memb_a_default(memb_a_default ptr);
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_a_cdecl' (aka 'void (NonVariadic::A::*)() __attribute__((cdecl))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_a_cdecl' (aka 'void (NonVariadic::A::*)() __attribute__((cdecl))') for 1st argument}}
void cb_memb_a_cdecl(memb_a_cdecl ptr);
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_a_thiscall' (aka 'void (NonVariadic::A::*)() __attribute__((thiscall))') for 1st argument}}
void cb_memb_a_thiscall(memb_a_thiscall ptr);
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_b_default' (aka 'void (NonVariadic::B::*)() __attribute__((thiscall))') for 1st argument}}
void cb_memb_b_default(memb_b_default ptr);
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_b_cdecl' (aka 'void (NonVariadic::B::*)() __attribute__((cdecl))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_b_cdecl' (aka 'void (NonVariadic::B::*)() __attribute__((cdecl))') for 1st argument}}
void cb_memb_b_cdecl(memb_b_cdecl ptr);
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_b_thiscall' (aka 'void (NonVariadic::B::*)() __attribute__((thiscall))') for 1st argument}}
void cb_memb_b_thiscall(memb_b_thiscall ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_c_default' (aka 'void (NonVariadic::C::*)() __attribute__((thiscall))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_c_default' (aka 'void (NonVariadic::C::*)() __attribute__((thiscall))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_c_default' (aka 'void (NonVariadic::C::*)() __attribute__((thiscall))') for 1st argument}}
void cb_memb_c_default(memb_c_default ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_c_cdecl' (aka 'void (NonVariadic::C::*)() __attribute__((cdecl))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_c_cdecl' (aka 'void (NonVariadic::C::*)() __attribute__((cdecl))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_c_cdecl' (aka 'void (NonVariadic::C::*)() __attribute__((cdecl))') for 1st argument}}
void cb_memb_c_cdecl(memb_c_cdecl ptr);
// expected-note@+3 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_c_thiscall' (aka 'void (NonVariadic::C::*)() __attribute__((thiscall))') for 1st argument}}
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((cdecl))' to 'NonVariadic::memb_c_thiscall' (aka 'void (NonVariadic::C::*)() __attribute__((thiscall))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (NonVariadic::A::*)() __attribute__((thiscall))' to 'NonVariadic::memb_c_thiscall' (aka 'void (NonVariadic::C::*)() __attribute__((thiscall))') for 1st argument}}
void cb_memb_c_thiscall(memb_c_thiscall ptr);

void call_member() {
  cb_memb_a_default(&A::member_default);
  cb_memb_a_default(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_a_default'}}
  cb_memb_a_default(&A::member_thiscall);

  cb_memb_a_cdecl(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_a_cdecl'}}
  cb_memb_a_cdecl(&A::member_cdecl);
  cb_memb_a_cdecl(&A::member_thiscall); // expected-error {{no matching function for call to 'cb_memb_a_cdecl'}}

  cb_memb_a_thiscall(&A::member_default);
  cb_memb_a_thiscall(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_a_thiscall'}}
  cb_memb_a_thiscall(&A::member_thiscall);
}

void call_member_inheritance() {
  cb_memb_b_default(&A::member_default);
  cb_memb_b_default(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_b_default'}}
  cb_memb_b_default(&A::member_thiscall);
  cb_memb_c_default(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_c_default'}}
  cb_memb_c_default(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_c_default'}}
  cb_memb_c_default(&A::member_thiscall); // expected-error {{no matching function for call to 'cb_memb_c_default'}}

  cb_memb_b_cdecl(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_b_cdecl'}}
  cb_memb_b_cdecl(&A::member_cdecl);
  cb_memb_b_cdecl(&A::member_thiscall); // expected-error {{no matching function for call to 'cb_memb_b_cdecl'}}
  cb_memb_c_cdecl(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_c_cdecl'}}
  cb_memb_c_cdecl(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_c_cdecl'}}
  cb_memb_c_cdecl(&A::member_thiscall); // expected-error {{no matching function for call to 'cb_memb_c_cdecl'}}

  cb_memb_b_thiscall(&A::member_default);
  cb_memb_b_thiscall(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_b_thiscall'}}
  cb_memb_b_thiscall(&A::member_thiscall);
  cb_memb_c_thiscall(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_c_thiscall'}}
  cb_memb_c_thiscall(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_c_thiscall'}}
  cb_memb_c_thiscall(&A::member_thiscall); // expected-error {{no matching function for call to 'cb_memb_c_thiscall'}}
}
} // end namespace NonVariadic

namespace Variadic {
struct A {
  void            member_default(int, ...);
  void __cdecl    member_cdecl(int, ...);
  void __thiscall member_thiscall(int, ...); // expected-error {{variadic function cannot use thiscall calling convention}}
};

struct B : public A {
};

struct C {
  void            member_default(int, ...);
  void __cdecl    member_cdecl(int, ...);
};

typedef void (           A::*memb_a_default)(int, ...);
typedef void (__cdecl    A::*memb_a_cdecl)(int, ...);
typedef void (           B::*memb_b_default)(int, ...);
typedef void (__cdecl    B::*memb_b_cdecl)(int, ...);
typedef void (           C::*memb_c_default)(int, ...);
typedef void (__cdecl    C::*memb_c_cdecl)(int, ...);

void cb_memb_a_default(memb_a_default ptr);
void cb_memb_a_cdecl(memb_a_cdecl ptr);
void cb_memb_b_default(memb_b_default ptr);
void cb_memb_b_cdecl(memb_b_cdecl ptr);
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (Variadic::A::*)(int, ...)' to 'Variadic::memb_c_default' (aka 'void (Variadic::C::*)(int, ...)') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (Variadic::A::*)(int, ...) __attribute__((cdecl))' to 'Variadic::memb_c_default' (aka 'void (Variadic::C::*)(int, ...)') for 1st argument}}
void cb_memb_c_default(memb_c_default ptr);
// expected-note@+2 {{candidate function not viable: no known conversion from 'void (Variadic::A::*)(int, ...)' to 'Variadic::memb_c_cdecl' (aka 'void (Variadic::C::*)(int, ...) __attribute__((cdecl))') for 1st argument}}
// expected-note@+1 {{candidate function not viable: no known conversion from 'void (Variadic::A::*)(int, ...) __attribute__((cdecl))' to 'Variadic::memb_c_cdecl' (aka 'void (Variadic::C::*)(int, ...) __attribute__((cdecl))') for 1st argument}}
void cb_memb_c_cdecl(memb_c_cdecl ptr);

void call_member() {
  cb_memb_a_default(&A::member_default);
  cb_memb_a_default(&A::member_cdecl);

  cb_memb_a_cdecl(&A::member_default);
  cb_memb_a_cdecl(&A::member_cdecl);
}

void call_member_inheritance() {
  cb_memb_b_default(&A::member_default);
  cb_memb_b_default(&A::member_cdecl);
  cb_memb_c_default(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_c_default'}}
  cb_memb_c_default(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_c_default'}}

  cb_memb_b_cdecl(&A::member_default);
  cb_memb_b_cdecl(&A::member_cdecl);
  cb_memb_c_cdecl(&A::member_default); // expected-error {{no matching function for call to 'cb_memb_c_cdecl'}}
  cb_memb_c_cdecl(&A::member_cdecl); // expected-error {{no matching function for call to 'cb_memb_c_cdecl'}}
}
} // end namespace Variadic

namespace MultiChunkDecls {

// Try to test declarators that have multiple DeclaratorChunks.
struct A {
  void __thiscall member_thiscall(int);
};

void (A::*return_mptr(short))(int) {
  return &A::member_thiscall;
}

void (A::*(*return_fptr_mptr(char))(short))(int) {
  return return_mptr;
}

typedef void (A::*mptr_t)(int);
mptr_t __stdcall return_mptr_std(short) {
  return &A::member_thiscall;
}

void (A::*(*return_fptr_std_mptr(char))(short))(int) {
  return return_mptr_std; // expected-error {{cannot initialize return object of type 'void (MultiChunkDecls::A::*(*)(short))(int) __attribute__((thiscall))' with an lvalue of type 'MultiChunkDecls::mptr_t (short) __attribute__((stdcall))'}}
}

void call_return() {
  A o;
  void (A::*(*fptr)(short))(int) = return_fptr_mptr('a');
  void (A::*mptr)(int) = fptr(1);
  (o.*mptr)(2);
}

} // end namespace MultiChunkDecls

namespace MemberPointers {

struct A {
  void __thiscall method_thiscall();
  void __cdecl    method_cdecl();
  void __stdcall  method_stdcall();
  void __fastcall method_fastcall();
};

void (           A::*mp1)() = &A::method_thiscall;
void (__cdecl    A::*mp2)() = &A::method_cdecl;
void (__stdcall  A::*mp3)() = &A::method_stdcall;
void (__fastcall A::*mp4)() = &A::method_fastcall;

// Use a typedef to form the member pointer and verify that cdecl is adjusted.
typedef void (           fun_default)();
typedef void (__cdecl    fun_cdecl)();
typedef void (__stdcall  fun_stdcall)();
typedef void (__fastcall fun_fastcall)();

fun_default  A::*td1 = &A::method_thiscall;
fun_cdecl    A::*td2 = &A::method_thiscall;
fun_stdcall  A::*td3 = &A::method_stdcall;
fun_fastcall A::*td4 = &A::method_fastcall;

// Round trip the function type through a template, and verify that only cdecl
// gets adjusted.
template<typename Fn> struct X { typedef Fn A::*p; };

X<void            ()>::p tmpl1 = &A::method_thiscall;
X<void __cdecl    ()>::p tmpl2 = &A::method_thiscall;
X<void __stdcall  ()>::p tmpl3 = &A::method_stdcall;
X<void __fastcall ()>::p tmpl4 = &A::method_fastcall;

X<fun_default >::p tmpl5 = &A::method_thiscall;
X<fun_cdecl   >::p tmpl6 = &A::method_thiscall;
X<fun_stdcall >::p tmpl7 = &A::method_stdcall;
X<fun_fastcall>::p tmpl8 = &A::method_fastcall;

// Make sure we adjust thiscall to cdecl when extracting the function type from
// a member pointer.
template <typename> struct Y;

template <typename Fn, typename C>
struct Y<Fn C::*> {
  typedef Fn *p;
};

void __cdecl f_cdecl();
Y<decltype(&A::method_thiscall)>::p tmpl9 = &f_cdecl;


} // end namespace MemberPointers

// Test that lambdas that capture nothing convert to cdecl function pointers.
namespace Lambdas {

void pass_fptr_cdecl   (void (__cdecl    *fp)());
void pass_fptr_stdcall (void (__stdcall  *fp)()); // expected-note {{candidate function not viable}}
void pass_fptr_fastcall(void (__fastcall *fp)()); // expected-note {{candidate function not viable}}

void conversion_to_fptr() {
  pass_fptr_cdecl   ([]() { } );
  pass_fptr_stdcall ([]() { } ); // expected-error {{no matching function for call}}
  pass_fptr_fastcall([]() { } ); // expected-error {{no matching function for call}}
}

}
