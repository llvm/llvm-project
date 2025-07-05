// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -fsyntax-only -verify=c,expected %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -fsyntax-only -verify=c,expected %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -fsyntax-only -x c++ -verify=cxx,expected %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -fsyntax-only -x c++ -verify=cxx,expected %s

#if defined __cplusplus
extern "C" {
#endif

void *memset(void *, int, __SIZE_TYPE__);
void bzero(void *, __SIZE_TYPE__);
void *memcpy(void *, const void *, __SIZE_TYPE__);
void *memmove(void *, const void *, __SIZE_TYPE__);

#if defined __cplusplus
}
#endif

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)

struct PtrAuthTrivial {
  int f0;
  int * IQ f1;
};

struct PtrAuthNonTrivial0 {
  int f0;
  int * AQ f1; // c-note 2 {{non-trivial to copy}}
  int f2;
};

struct PtrAuthNonTrivial1 {
  int * AQ f0; // c-note 2 {{non-trivial to copy}}
  int f1;
  struct PtrAuthNonTrivial0 f2;
};

void testPtrAuthTrivial(struct PtrAuthTrivial *d, struct PtrAuthTrivial *s) {
  memset(d, 0, sizeof(struct PtrAuthTrivial));
  memset(d, 1, sizeof(struct PtrAuthTrivial));
  bzero(d, sizeof(struct PtrAuthTrivial));
  memcpy(d, s, sizeof(struct PtrAuthTrivial));
  memmove(d, s, sizeof(struct PtrAuthTrivial));
}

void testPtrAuthNonTrivial1(struct PtrAuthNonTrivial1 *d,
                            struct PtrAuthNonTrivial1 *s) {
  memset(d, 0, sizeof(struct PtrAuthNonTrivial1)); // cxx-warning {{is a pointer to non-trivially copyable type 'struct PtrAuthNonTrivial1'}} // cxx-note {{explicitly cast the pointer to silence}}
  memset(d, 1, sizeof(struct PtrAuthNonTrivial1)); // cxx-warning {{is a pointer to non-trivially copyable type 'struct PtrAuthNonTrivial1'}} // cxx-note {{explicitly cast the pointer to silence}}
  bzero(d, sizeof(struct PtrAuthNonTrivial1)); // cxx-warning {{is a pointer to non-trivially copyable type 'struct PtrAuthNonTrivial1'}} // cxx-note {{explicitly cast the pointer to silence}}
  memcpy(d, s, sizeof(struct PtrAuthNonTrivial1));
  // c-warning@-1 {{that is not trivial to primitive-copy}}
  // cxx-warning@-2 {{is a pointer to non-trivially copyable type 'struct PtrAuthNonTrivial1'}}
  // expected-note@-3 {{explicitly cast the pointer to silence}}
  memmove(d, s, sizeof(struct PtrAuthNonTrivial1));
  // c-warning@-1 {{that is not trivial to primitive-copy}}
  // cxx-warning@-2 {{is a pointer to non-trivially copyable type 'struct PtrAuthNonTrivial1'}}
  // expected-note@-3 {{explicitly cast the pointer to silence}}
}
