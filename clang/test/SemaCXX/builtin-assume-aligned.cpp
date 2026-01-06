// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx11 -std=c++11 -triple x86_64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx11 -std=c++11 -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx26 -std=c++26 -triple x86_64-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx26 -std=c++26 -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter %s

int n;
constexpr int *p = 0;
// expected-error@+2 {{must be initialized by a constant expression}}
// cxx26-note@+1 {{a constant expression cannot modify an object that is visible outside that expression}}
constexpr int *k = (int *) __builtin_assume_aligned(p, 16, n = 5);

constexpr void *l = __builtin_assume_aligned(p, 16);

// cxx11-error@+2 {{must be initialized by a constant expression}}
// cxx11-note@+1 {{cast from 'void *' is not allowed in a constant expression}}
constexpr int *c = (int *) __builtin_assume_aligned(p, 16);

// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{alignment of the base pointee object (4 bytes) is less than the asserted 16 bytes}}
constexpr void *m = __builtin_assume_aligned(&n, 16);

// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (-2 bytes) is not a multiple of the asserted 4 bytes}}
constexpr void *q1 = __builtin_assume_aligned(&n, 4, 2);
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (2 bytes) is not a multiple of the asserted 4 bytes}}
constexpr void *q2 = __builtin_assume_aligned(&n, 4, -2);
constexpr void *q3 = __builtin_assume_aligned(&n, 4, 4);
constexpr void *q4 = __builtin_assume_aligned(&n, 4, -4);

static char ar1[6];
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{alignment of the base pointee object (1 byte) is less than the asserted 16 bytes}}
constexpr void *r1 = __builtin_assume_aligned(&ar1[2], 16);

static char ar2[6] __attribute__((aligned(32)));
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (2 bytes) is not a multiple of the asserted 16 bytes}}
constexpr void *r2 = __builtin_assume_aligned(&ar2[2], 16);
constexpr void *r3 = __builtin_assume_aligned(&ar2[2], 16, 2);
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{offset of the aligned pointer from the base pointee object (1 byte) is not a multiple of the asserted 16 bytes}}
constexpr void *r4 = __builtin_assume_aligned(&ar2[2], 16, 1);

constexpr int* x = __builtin_constant_p((int*)0xFF) ? (int*)0xFF : (int*)0xFF;
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{value of the aligned pointer (255) is not a multiple of the asserted 32 bytes}}
constexpr void *s1 = __builtin_assume_aligned(x, 32);
// expected-error@+2 {{must be initialized by a constant expression}}
// expected-note@+1 {{value of the aligned pointer (250) is not a multiple of the asserted 32 bytes}}
constexpr void *s2 = __builtin_assume_aligned(x, 32, 5);
constexpr void *s3 = __builtin_assume_aligned(x, 32, -1);


constexpr int add(int a, int b) {
  return a+b;
}
constexpr void *c1 = __builtin_assume_aligned(p, add(1,1));
constexpr void *c2 = __builtin_assume_aligned(p, add(2,1)); // expected-error {{not a power of 2}}

constexpr long kAlignment = 128;
long AllocateAlignedBytes_payload;
void AllocateAlignedBytes() {
  void *m = __builtin_assume_aligned(
      reinterpret_cast<void *>(AllocateAlignedBytes_payload), kAlignment);
}

namespace std {
  enum class byte : unsigned char {};
} // namespace std

namespace GH173767 {
#if __cplusplus > 202302L
  constexpr int a1a = (delete[] (unsigned char*)__builtin_assume_aligned(new unsigned char[65], __STDCPP_DEFAULT_NEW_ALIGNMENT__), 0);
  constexpr int a1b = (delete[] (char*)__builtin_assume_aligned(new char[65], __STDCPP_DEFAULT_NEW_ALIGNMENT__), 0);
  constexpr int a1c = (delete[] (std::byte*)__builtin_assume_aligned(new std::byte[65], __STDCPP_DEFAULT_NEW_ALIGNMENT__), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (16 bytes) is less than the asserted 32 bytes}}
  constexpr int a2a = (delete[] (unsigned char*)__builtin_assume_aligned(new unsigned char[65], 2*__STDCPP_DEFAULT_NEW_ALIGNMENT__), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (16 bytes) is less than the asserted 32 bytes}}
  constexpr int a2b = (delete[] (char*)__builtin_assume_aligned(new char[65], 2*__STDCPP_DEFAULT_NEW_ALIGNMENT__), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (16 bytes) is less than the asserted 32 bytes}}
  constexpr int a2c = (delete[] (std::byte*)__builtin_assume_aligned(new std::byte[65], 2*__STDCPP_DEFAULT_NEW_ALIGNMENT__), 0);

  constexpr int b1 = (delete (int*)__builtin_assume_aligned(new int, alignof(int)), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (4 bytes) is less than the asserted 8 bytes}}
  constexpr int b2 = (delete (int*)__builtin_assume_aligned(new int, 2*alignof(int)), 0);

  constexpr int c1 = (delete[] (int*)__builtin_assume_aligned(new int[4], alignof(int)), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (4 bytes) is less than the asserted 8 bytes}}
  constexpr int c2 = (delete[] (int*)__builtin_assume_aligned(new int[4], 2*alignof(int)), 0);

  struct D {
    alignas(2*__STDCPP_DEFAULT_NEW_ALIGNMENT__) int x[2];
  };

  constexpr int d1 = (delete (D*)__builtin_assume_aligned(new D, alignof(D)), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (32 bytes) is less than the asserted 64 bytes}}
  constexpr int d2 = (delete (D*)__builtin_assume_aligned(new D, 2*alignof(D)), 0);

  constexpr int d3 = []{
    auto p = new D;
    (void)__builtin_assume_aligned(p->x + 1, alignof(int));
    delete p;
    return 0;
  }();

  // expected-error@+3 {{must be initialized by a constant expression}}
  // expected-note@+2 {{in call to}}
  // expected-note@+3 {{offset of the aligned pointer from the base pointee object (4 bytes) is not a multiple of the asserted 8 bytes}}
  constexpr int d4 = []{
    auto p = new D;
    (void)__builtin_assume_aligned(p->x + 1, 2*alignof(int));
    delete p;
    return 0;
  }();

  struct E {
    unsigned char x[65];
  };

  constexpr int e1 = (delete (E*)__builtin_assume_aligned(new E, 1), 0);
  // expected-error@+2 {{must be initialized by a constant expression}}
  // expected-note@+1 {{alignment of the base pointee object (1 byte) is less than the asserted 2 bytes}}
  constexpr int e2 = (delete (E*)__builtin_assume_aligned(new E, 2), 0);

  constexpr int f = []{
    auto p = new int;
    delete p;
    (void)__builtin_assume_aligned(p, alignof(int));
    return 0;
  }();
#endif
} // namespace GH173767
