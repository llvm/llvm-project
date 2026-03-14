// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin9 %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-linux-gnu %s -DHAVE_NOT
// RUN: %clang_cc1 -fsyntax-only -verify -triple powerpc64-ibm-aix-xcoff %s -DHAVE
// RUN: %clang_cc1 -fsyntax-only -verify -triple powerpc-ibm-aix-xcoff %s -DHAVE_NOT

#ifdef HAVE
typedef int i128 __attribute__((__mode__(TI)));
typedef unsigned u128 __attribute__((__mode__(TI)));

int a[((i128)-1 ^ (i128)-2) == 1 ? 1 : -1];
int a[(u128)-1 > 1LL ? 1 : -1];
int a[__SIZEOF_INT128__ == 16 ? 1 : -1];

// PR5435
__uint128_t b = (__uint128_t)-1;

// PR11916: Support for libstdc++ 4.7
__int128 i = (__int128)0;
unsigned __int128 u = (unsigned __int128)-1;

long long SignedTooBig = 123456789012345678901234567890; // expected-error {{integer literal is too large to be represented in any integer type}}
unsigned long long UnsignedTooBig = 123456789012345678901234567890; // expected-error {{integer literal is too large to be represented in any integer type}}

void MPI_Send(void *buf, int datatype) __attribute__(( pointer_with_type_tag(mpi,1,2) ));

static const int mpi_int __attribute__(( type_tag_for_datatype(mpi,int) )) = 10;

void test(int *buf)
{
}
#else

__int128 n; // expected-error {{__int128 is not supported on this target}}

#if defined(__SIZEOF_INT128__)
#error __SIZEOF_INT128__ should not be defined
#endif

#endif
