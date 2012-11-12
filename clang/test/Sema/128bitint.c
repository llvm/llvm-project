// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin9 -fms-extensions %s
typedef int i128 __attribute__((__mode__(TI)));
typedef unsigned u128 __attribute__((__mode__(TI)));

int a[((i128)-1 ^ (i128)-2) == 1 ? 1 : -1];
int a[(u128)-1 > 1LL ? 1 : -1];

// PR5435
__uint128_t b = (__uint128_t)-1;

// PR11916: Support for libstdc++ 4.7
__int128 i = (__int128)0;
unsigned __int128 u = (unsigned __int128)-1;

long long SignedTooBig = 123456789012345678901234567890; // expected-warning {{integer constant is too large for its type}}
__int128_t Signed128 = 123456789012345678901234567890i128;
long long Signed64 = 123456789012345678901234567890i128; // expected-warning {{implicit conversion from '__int128' to 'long long' changes value from 123456789012345678901234567890 to -4362896299872285998}}
unsigned long long UnsignedTooBig = 123456789012345678901234567890; // expected-warning {{integer constant is too large for its type}}
__uint128_t Unsigned128 = 123456789012345678901234567890Ui128;
unsigned long long Unsigned64 = 123456789012345678901234567890Ui128; // expected-warning {{implicit conversion from 'unsigned __int128' to 'unsigned long long' changes value from 123456789012345678901234567890 to 14083847773837265618}}

// Ensure we don't crash when user passes 128-bit values to type safety
// attributes.
void pointer_with_type_tag_arg_num_1(void *buf, int datatype)
    __attribute__(( pointer_with_type_tag(mpi,0x10000000000000001i128,1) )); // expected-error {{attribute parameter 2 is out of bounds}}

void pointer_with_type_tag_arg_num_2(void *buf, int datatype)
    __attribute__(( pointer_with_type_tag(mpi,1,0x10000000000000001i128) )); // expected-error {{attribute parameter 3 is out of bounds}}

void MPI_Send(void *buf, int datatype) __attribute__(( pointer_with_type_tag(mpi,1,2) ));

static const __uint128_t mpi_int_wrong __attribute__(( type_tag_for_datatype(mpi,int) )) = 0x10000000000000001i128; // expected-error {{'type_tag_for_datatype' attribute requires the initializer to be an integer constant expression that can be represented by a 64 bit integer}}
static const int mpi_int __attribute__(( type_tag_for_datatype(mpi,int) )) = 10;

void test(int *buf)
{
  MPI_Send(buf, 0x10000000000000001i128); // expected-warning {{implicit conversion from '__int128' to 'int' changes value}}
}

