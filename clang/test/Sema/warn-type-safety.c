// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -std=c++98 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c99 -fno-signed-char -fsyntax-only -verify %s

struct A {};

typedef struct A *MPI_Datatype;

int wrong1(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag )); // expected-error {{'pointer_with_type_tag' attribute requires exactly 3 arguments}}

int wrong2(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,0,7) )); // expected-error {{attribute parameter 2 is out of bounds}}

int wrong3(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,3,7) )); // expected-error {{attribute parameter 2 is out of bounds}}

int wrong4(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,1,0) )); // expected-error {{attribute parameter 3 is out of bounds}}

int wrong5(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,1,3) )); // expected-error {{attribute parameter 3 is out of bounds}}

int wrong6(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,0x8000000000000001ULL,1) )); // expected-error {{attribute parameter 2 is out of bounds}}

extern int x;

int wrong7(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,x,2) )); // expected-error {{attribute requires parameter 2 to be an integer constant}}

int wrong8(void *buf, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,1,x) )); // expected-error {{attribute requires parameter 3 to be an integer constant}}

int wrong9 __attribute__(( pointer_with_type_tag(mpi,1,2) )); // expected-error {{'pointer_with_type_tag' attribute only applies to non-K&R-style functions}}

int wrong10(double buf, MPI_Datatype type)
    __attribute__(( pointer_with_type_tag(mpi,1,2) )); // expected-error {{'pointer_with_type_tag' attribute only applies to pointer arguments}}

int ok11(void *, ...)
    __attribute__(( pointer_with_type_tag(mpi,1,2) ));
int wrong11(void *, ...)
    __attribute__(( pointer_with_type_tag(mpi,2,3) )); // expected-error {{'pointer_with_type_tag' attribute only applies to pointer arguments}}

extern struct A datatype_wrong1
    __attribute__(( type_tag_for_datatype )); // expected-error {{'type_tag_for_datatype' attribute requires parameter 1 to be an identifier}}

extern struct A datatype_wrong2
    __attribute__(( type_tag_for_datatype(mpi,1,2) )); // expected-error {{expected a type}}

extern struct A datatype_wrong3
    __attribute__(( type_tag_for_datatype(mpi,not_a_type) )); // expected-error {{unknown type name 'not_a_type'}}

extern struct A datatype_wrong4
    __attribute__(( type_tag_for_datatype(mpi,int,int) )); // expected-error {{expected identifier}}

extern struct A datatype_wrong5
    __attribute__(( type_tag_for_datatype(mpi,int,not_a_flag) )); // expected-error {{invalid comparison flag 'not_a_flag'}}

extern struct A datatype_wrong6
    __attribute__(( type_tag_for_datatype(mpi,int,layout_compatible,not_a_flag) )); // expected-error {{invalid comparison flag 'not_a_flag'}}


void datatype_wrong7(void) __attribute__((type_tag_for_datatype(datatype_wrong7, int))); // expected-error {{'type_tag_for_datatype' attribute only applies to variables}}

// Using a tag with kind A in a place where the function requires kind B should
// warn.

void A_func(void *ptr, void *tag) __attribute__(( pointer_with_type_tag(a,1,2) ));

extern struct A A_tag __attribute__(( type_tag_for_datatype(a,int) ));
extern struct A B_tag __attribute__(( type_tag_for_datatype(b,int) ));

void C_func(void *ptr, int tag) __attribute__(( pointer_with_type_tag(c,1,2) ));

static const int C_tag __attribute__(( type_tag_for_datatype(c,int) )) = 10;
static const int D_tag __attribute__(( type_tag_for_datatype(d,int) )) = 20;

void test_tag_mismatch(int *ptr)
{
  A_func(ptr, &A_tag); // no-warning
  A_func(ptr, &B_tag); // expected-warning {{this type tag was not designed to be used with this function}}
  C_func(ptr, C_tag); // no-warning
  C_func(ptr, D_tag); // expected-warning {{this type tag was not designed to be used with this function}}
  C_func(ptr, 10); // no-warning
  C_func(ptr, 20); // should warn, but may cause false positives
}

void test_null_pointer()
{
  C_func(0, C_tag); // no-warning
  C_func((void *) 0, C_tag); // no-warning
  C_func((int *) 0, C_tag); // no-warning
  C_func((long *) 0, C_tag); // expected-warning {{argument type 'long *' doesn't match specified 'c' type tag that requires 'int *'}}
}

// Check that we look through typedefs in the special case of allowing 'char'
// to be matched with 'signed char' or 'unsigned char'.
void E_func(void *ptr, int tag) __attribute__(( pointer_with_type_tag(e,1,2) ));

typedef char E_char;
typedef char E_char_2;
typedef signed char E_char_signed;
typedef unsigned char E_char_unsigned;

static const int E_tag __attribute__(( type_tag_for_datatype(e,E_char) )) = 10;

void test_char_typedef(char *char_buf,
                       E_char_2 *e_char_buf,
                       E_char_signed *e_char_signed_buf,
                       E_char_unsigned *e_char_unsigned_buf)
{
  E_func(char_buf, E_tag);
  E_func(e_char_buf, E_tag);
#ifdef __CHAR_UNSIGNED__
  E_func(e_char_signed_buf, E_tag); // expected-warning {{argument type 'E_char_signed *' (aka 'signed char *') doesn't match specified 'e' type tag that requires 'E_char *' (aka 'char *')}}
  E_func(e_char_unsigned_buf, E_tag);
#else
  E_func(e_char_signed_buf, E_tag);
  E_func(e_char_unsigned_buf, E_tag); // expected-warning {{argument type 'E_char_unsigned *' (aka 'unsigned char *') doesn't match specified 'e' type tag that requires 'E_char *' (aka 'char *')}}
#endif
}

// Tests for argument_with_type_tag.

#define F_DUPFD 10
#define F_SETLK 20

struct flock { };

static const int F_DUPFD_tag __attribute__(( type_tag_for_datatype(fcntl,int) )) = F_DUPFD;
static const int F_SETLK_tag __attribute__(( type_tag_for_datatype(fcntl,struct flock *) )) = F_SETLK;

int fcntl(int fd, int cmd, ...) __attribute__(( argument_with_type_tag(fcntl,3,2) ));

void test_argument_with_type_tag(struct flock *f)
{
  fcntl(0, F_DUPFD, 10); // no-warning
  fcntl(0, F_SETLK, f);  // no-warning

  fcntl(0, F_SETLK, 10); // expected-warning {{argument type 'int' doesn't match specified 'fcntl' type tag that requires 'struct flock *'}}
  fcntl(0, F_DUPFD, f);  // expected-warning {{argument type 'struct flock *' doesn't match specified 'fcntl' type tag that requires 'int'}}
}

void test_tag_expresssion(int b) {
  fcntl(0, b ? F_DUPFD : F_SETLK, 10); // no-warning
  fcntl(0, b + F_DUPFD, 10); // no-warning
  fcntl(0, (b, F_DUPFD), 10); // expected-warning {{left operand of comma operator has no effect}}
}

// Check that using 64-bit magic values as tags works and tag values do not
// overflow internally.
void F_func(void *ptr, unsigned long long tag) __attribute__((pointer_with_type_tag(f,1,2) ));

static const unsigned long long F_tag1 __attribute__(( type_tag_for_datatype(f,int) )) = 0xFFFFFFFFFFFFFFFFULL;
static const unsigned long long F_tag2 __attribute__(( type_tag_for_datatype(f,float) )) = 0xFFFFFFFFULL;

void test_64bit_magic(int *int_ptr, float *float_ptr)
{
  F_func(int_ptr,   0xFFFFFFFFFFFFFFFFULL);
  F_func(int_ptr,   0xFFFFFFFFULL);         // expected-warning {{argument type 'int *' doesn't match specified 'f' type tag that requires 'float *'}}
  F_func(float_ptr, 0xFFFFFFFFFFFFFFFFULL); // expected-warning {{argument type 'float *' doesn't match specified 'f' type tag that requires 'int *'}}
  F_func(float_ptr, 0xFFFFFFFFULL);
}
