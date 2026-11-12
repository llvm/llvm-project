// RUN: %clang_cc1 -std=c99 -DOPEN_MPI -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c99 -DMPICH -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -std=c++98 -DOPEN_MPI -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -std=c++98 -DMPICH -fsyntax-only -verify %s
//
// RUN: %clang_cc1 -std=c99 -DOPEN_MPI -fno-signed-char -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c99 -DMPICH -fno-signed-char -fsyntax-only -verify %s

//===--- limits.h mock ----------------------------------------------------===//

#ifdef __CHAR_UNSIGNED__
#define CHAR_MIN 0
#define CHAR_MAX (__SCHAR_MAX__*2  +1)
#else
#define CHAR_MIN (-__SCHAR_MAX__-1)
#define CHAR_MAX __SCHAR_MAX__
#endif

//===--- mpi.h mock -------------------------------------------------------===//

#define NULL ((void *)0)

#ifdef OPEN_MPI
typedef struct ompi_datatype_t *MPI_Datatype;
#endif

#ifdef MPICH
typedef int MPI_Datatype;
#endif

int MPI_Send(void *buf, int count, MPI_Datatype datatype)
    __attribute__(( pointer_with_type_tag(mpi,1,3) ));

int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype)
               __attribute__(( pointer_with_type_tag(mpi,1,3), pointer_with_type_tag(mpi,4,6) ));

#ifdef OPEN_MPI
// OpenMPI and LAM/MPI-style datatype definitions

#define OMPI_PREDEFINED_GLOBAL(type, global) ((type) &(global))

#define MPI_DATATYPE_NULL OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_datatype_null)
#define MPI_FLOAT         OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_float)
#define MPI_INT           OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_int)
#define MPI_LONG          OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_long)
#define MPI_LONG_LONG_INT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_long_long_int)
#define MPI_CHAR          OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_char)

#define MPI_FLOAT_INT     OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_float_int)
#define MPI_2INT          OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_2int)

#define MPI_IN_PLACE ((void *) 1)

extern struct ompi_predefined_datatype_t ompi_mpi_datatype_null __attribute__(( type_tag_for_datatype(mpi,void,must_be_null) ));
extern struct ompi_predefined_datatype_t ompi_mpi_float         __attribute__(( type_tag_for_datatype(mpi,float) ));
extern struct ompi_predefined_datatype_t ompi_mpi_int           __attribute__(( type_tag_for_datatype(mpi,int) ));
extern struct ompi_predefined_datatype_t ompi_mpi_long          __attribute__(( type_tag_for_datatype(mpi,long) ));
extern struct ompi_predefined_datatype_t ompi_mpi_long_long_int __attribute__(( type_tag_for_datatype(mpi,long long int) ));
extern struct ompi_predefined_datatype_t ompi_mpi_char          __attribute__(( type_tag_for_datatype(mpi,char) ));

struct ompi_struct_mpi_float_int {float f; int i;};
extern struct ompi_predefined_datatype_t ompi_mpi_float_int     __attribute__(( type_tag_for_datatype(mpi, struct ompi_struct_mpi_float_int, layout_compatible) ));

struct ompi_struct_mpi_2int {int i1; int i2;};
extern struct ompi_predefined_datatype_t ompi_mpi_2int          __attribute__(( type_tag_for_datatype(mpi, struct ompi_struct_mpi_2int, layout_compatible) ));
#endif

#ifdef MPICH
// MPICH2 and MVAPICH2-style datatype definitions

#define MPI_COMM_WORLD ((MPI_Comm) 0x44000000)

#define MPI_DATATYPE_NULL ((MPI_Datatype) 0xa0000000)
#define MPI_FLOAT         ((MPI_Datatype) 0xa0000001)
#define MPI_INT           ((MPI_Datatype) 0xa0000002)
#define MPI_LONG          ((MPI_Datatype) 0xa0000003)
#define MPI_LONG_LONG_INT ((MPI_Datatype) 0xa0000004)
#define MPI_CHAR          ((MPI_Datatype) 0xa0000005)

#define MPI_FLOAT_INT     ((MPI_Datatype) 0xa0000006)
#define MPI_2INT          ((MPI_Datatype) 0xa0000007)

#define MPI_IN_PLACE  (void *) -1

static const MPI_Datatype mpich_mpi_datatype_null __attribute__(( type_tag_for_datatype(mpi,void,must_be_null) )) = 0xa0000000;
static const MPI_Datatype mpich_mpi_float         __attribute__(( type_tag_for_datatype(mpi,float) ))             = 0xa0000001;
static const MPI_Datatype mpich_mpi_int           __attribute__(( type_tag_for_datatype(mpi,int) ))               = 0xa0000002;
static const MPI_Datatype mpich_mpi_long          __attribute__(( type_tag_for_datatype(mpi,long) ))              = 0xa0000003;
static const MPI_Datatype mpich_mpi_long_long_int __attribute__(( type_tag_for_datatype(mpi,long long int) ))     = 0xa0000004;
static const MPI_Datatype mpich_mpi_char          __attribute__(( type_tag_for_datatype(mpi,char) ))              = 0xa0000005;

struct mpich_struct_mpi_float_int { float f; int i; };
struct mpich_struct_mpi_2int { int i1; int i2; };
static const MPI_Datatype mpich_mpi_float_int     __attribute__(( type_tag_for_datatype(mpi, struct mpich_struct_mpi_float_int, layout_compatible) )) = 0xa0000006;
static const MPI_Datatype mpich_mpi_2int          __attribute__(( type_tag_for_datatype(mpi, struct mpich_struct_mpi_2int, layout_compatible) )) = 0xa0000007;
#endif

//===--- HDF5 headers mock ------------------------------------------------===//

typedef int hid_t;
void H5open(void);

#ifndef HDF_PRIVATE
#define H5OPEN  H5open(),
#else
#define H5OPEN
#endif

#define H5T_NATIVE_CHAR         (CHAR_MIN?H5T_NATIVE_SCHAR:H5T_NATIVE_UCHAR)
#define H5T_NATIVE_SCHAR        (H5OPEN H5T_NATIVE_SCHAR_g)
#define H5T_NATIVE_UCHAR        (H5OPEN H5T_NATIVE_UCHAR_g)
#define H5T_NATIVE_INT          (H5OPEN H5T_NATIVE_INT_g)
#define H5T_NATIVE_LONG         (H5OPEN H5T_NATIVE_LONG_g)

hid_t H5T_NATIVE_SCHAR_g __attribute__(( type_tag_for_datatype(hdf5,signed char) ));
hid_t H5T_NATIVE_UCHAR_g __attribute__(( type_tag_for_datatype(hdf5,unsigned char) ));
hid_t H5T_NATIVE_INT_g   __attribute__(( type_tag_for_datatype(hdf5,int) ));
hid_t H5T_NATIVE_LONG_g  __attribute__(( type_tag_for_datatype(hdf5,long) ));

void H5Dwrite(hid_t mem_type_id, const void *buf) __attribute__(( pointer_with_type_tag(hdf5,2,1) ));

//===--- Tests ------------------------------------------------------------===//

//===--- MPI

struct pair_float_int
{
  float f; int i;
};

struct pair_int_int
{
  int i1; int i2;
};

void test_mpi_predefined_types(
    int *int_buf,
    long *long_buf1,
    long *long_buf2,
    void *void_buf,
    struct pair_float_int *pfi,
    struct pair_int_int *pii)
{
  char char_buf[255];

  // Layout-compatible scalar types.
  MPI_Send(int_buf,   1, MPI_INT); // no-warning

  // Null pointer constant.
  MPI_Send(0,         0, MPI_INT); // no-warning
  MPI_Send(NULL,      0, MPI_INT); // no-warning

  // Layout-compatible class types.
  MPI_Send(pfi, 1, MPI_FLOAT_INT); // no-warning
  MPI_Send(pii, 1, MPI_2INT); // no-warning

  // Layout-incompatible scalar types.
  MPI_Send(long_buf1, 1, MPI_INT); // expected-warning {{argument type 'long *' doesn't match specified 'mpi' type tag that requires 'int *'}}

  // Layout-incompatible class types.
  MPI_Send(pii, 1, MPI_FLOAT_INT); // expected-warning {{argument type 'struct pair_int_int *' doesn't match specified 'mpi' type tag}}
  MPI_Send(pfi, 1, MPI_2INT); // expected-warning {{argument type 'struct pair_float_int *' doesn't match specified 'mpi' type tag}}

  // Layout-incompatible class-scalar types.
  MPI_Send(long_buf1, 1, MPI_2INT); // expected-warning {{argument type 'long *' doesn't match specified 'mpi' type tag}}

  // Function with two buffers.
  MPI_Gather(long_buf1, 1, MPI_INT,  // expected-warning {{argument type 'long *' doesn't match specified 'mpi' type tag that requires 'int *'}}
             long_buf2, 1, MPI_INT); // expected-warning {{argument type 'long *' doesn't match specified 'mpi' type tag that requires 'int *'}}

  // Array buffers should work like pointer buffers.
  MPI_Send(char_buf,  255, MPI_CHAR); // no-warning

  // Explicit casts should not be dropped.
  MPI_Send((int *) char_buf,  255, MPI_INT); // no-warning
  MPI_Send((int *) char_buf,  255, MPI_CHAR); // expected-warning {{argument type 'int *' doesn't match specified 'mpi' type tag that requires 'char *'}}

  // `void*' buffer should never warn.
  MPI_Send(void_buf,  255, MPI_CHAR); // no-warning

  // We expect that MPI_IN_PLACE is `void*', shouldn't warn.
  MPI_Gather(MPI_IN_PLACE, 0, MPI_INT,
             int_buf,      1, MPI_INT);

  // Special handling for MPI_DATATYPE_NULL: buffer pointer should be either
  // a `void*' pointer or a null pointer constant.
  MPI_Gather(NULL,    0, MPI_DATATYPE_NULL, // no-warning
             int_buf, 1, MPI_INT);

  MPI_Gather(int_buf, 0, MPI_DATATYPE_NULL, // expected-warning {{specified mpi type tag requires a null pointer}}
             int_buf, 1, MPI_INT);
}

MPI_Datatype my_int_datatype __attribute__(( type_tag_for_datatype(mpi,int) ));

struct S1 { int a; int b; };
MPI_Datatype my_s1_datatype __attribute__(( type_tag_for_datatype(mpi,struct S1) ));

// Layout-compatible to S1, but should be treated as a different type.
struct S2 { int a; int b; };
MPI_Datatype my_s2_datatype __attribute__(( type_tag_for_datatype(mpi,struct S2) ));

enum E1 { Foo };
MPI_Datatype my_e1_datatype __attribute__(( type_tag_for_datatype(mpi,enum E1) ));

void test_user_types(int *int_buf,
                     long *long_buf,
                     struct S1 *s1_buf,
                     struct S2 *s2_buf,
                     enum E1 *e1_buf)
{
  MPI_Send(int_buf,  1, my_int_datatype); // no-warning
  MPI_Send(long_buf, 1, my_int_datatype); // expected-warning {{argument type 'long *' doesn't match specified 'mpi' type tag that requires 'int *'}}

  MPI_Send(s1_buf, 1, my_s1_datatype); // no-warning
  MPI_Send(s1_buf, 1, my_s2_datatype); // expected-warning {{argument type 'struct S1 *' doesn't match specified 'mpi' type tag that requires 'struct S2 *'}}

  MPI_Send(long_buf, 1, my_s1_datatype); // expected-warning {{argument type 'long *' doesn't match specified 'mpi' type tag that requires 'struct S1 *'}}
  MPI_Send(s1_buf, 1, MPI_INT); // expected-warning {{argument type 'struct S1 *' doesn't match specified 'mpi' type tag that requires 'int *'}}

  MPI_Send(e1_buf, 1, my_e1_datatype); // no-warning
  MPI_Send(e1_buf, 1, MPI_INT); // expected-warning {{argument type 'enum E1 *' doesn't match specified 'mpi' type tag that requires 'int *'}}
  MPI_Send(int_buf, 1, my_e1_datatype); // expected-warning {{argument type 'int *' doesn't match specified 'mpi' type tag that requires 'enum E1 *'}}
}

MPI_Datatype my_unknown_datatype;

void test_not_annotated(int *int_buf,
                        long *long_buf,
                        MPI_Datatype type)
{
  // Using 'MPI_Datatype's without attributes should not produce warnings.
  MPI_Send(long_buf, 1, my_unknown_datatype); // no-warning
  MPI_Send(int_buf, 1, type); // no-warning
}

struct S1_compat { int a; int b; };
MPI_Datatype my_s1_compat_datatype
    __attribute__(( type_tag_for_datatype(mpi, struct S1_compat, layout_compatible) ));

struct S3        { int a; long b; double c; double d; struct S1 s1; };
struct S3_compat { int a; long b; double c; double d; struct S2 s2; };
MPI_Datatype my_s3_compat_datatype
    __attribute__(( type_tag_for_datatype(mpi, struct S3_compat, layout_compatible) ));

struct S4        { char c; };
struct S4_compat { signed char c; };
MPI_Datatype my_s4_compat_datatype
    __attribute__(( type_tag_for_datatype(mpi, struct S4_compat, layout_compatible) ));

union U1        { int a; long b; double c; double d; struct S1 s1; };
union U1_compat { long b; double c; struct S2 s; int a; double d; };
MPI_Datatype my_u1_compat_datatype
    __attribute__(( type_tag_for_datatype(mpi, union U1_compat, layout_compatible) ));

union U2 { int a; long b; double c; struct S1 s1; };
MPI_Datatype my_u2_datatype
    __attribute__(( type_tag_for_datatype(mpi, union U2, layout_compatible) ));

void test_layout_compatibility(struct S1 *s1_buf, struct S3 *s3_buf,
                               struct S4 *s4_buf,
                               union U1 *u1_buf, union U2 *u2_buf)
{
  MPI_Send(s1_buf, 1, my_s1_compat_datatype); // no-warning
  MPI_Send(s3_buf, 1, my_s3_compat_datatype); // no-warning
  MPI_Send(s1_buf, 1, my_s3_compat_datatype); // expected-warning {{argument type 'struct S1 *' doesn't match specified 'mpi' type tag}}
  MPI_Send(s4_buf, 1, my_s4_compat_datatype); // expected-warning {{argument type 'struct S4 *' doesn't match specified 'mpi' type tag}}
  MPI_Send(u1_buf, 1, my_u1_compat_datatype); // no-warning
  MPI_Send(u1_buf, 1, my_u2_datatype);        // expected-warning {{argument type 'union U1 *' doesn't match specified 'mpi' type tag}}
  MPI_Send(u2_buf, 1, my_u1_compat_datatype); // expected-warning {{argument type 'union U2 *' doesn't match specified 'mpi' type tag}}
}

// There is an MPI_REAL predefined in MPI, but some existing MPI programs do
// this.
typedef float real;
#define MPI_REAL MPI_FLOAT

void test_mpi_real_user_type(real *real_buf, float *float_buf)
{
  MPI_Send(real_buf,  1, MPI_REAL);  // no-warning
  MPI_Send(real_buf,  1, MPI_FLOAT); // no-warning
  MPI_Send(float_buf, 1, MPI_REAL);  // no-warning
  MPI_Send(float_buf, 1, MPI_FLOAT); // no-warning
}

//===--- HDF5

void test_hdf5(char *char_buf,
               signed char *schar_buf,
               unsigned char *uchar_buf,
               int *int_buf,
               long *long_buf)
{
  H5Dwrite(H5T_NATIVE_CHAR,  char_buf);  // no-warning
#ifdef __CHAR_UNSIGNED__
  H5Dwrite(H5T_NATIVE_CHAR,  schar_buf); // expected-warning {{argument type 'signed char *' doesn't match specified 'hdf5' type tag that requires 'unsigned char *'}}
  H5Dwrite(H5T_NATIVE_CHAR,  uchar_buf); // no-warning
#else
  H5Dwrite(H5T_NATIVE_CHAR,  schar_buf); // no-warning
  H5Dwrite(H5T_NATIVE_CHAR,  uchar_buf); // expected-warning {{argument type 'unsigned char *' doesn't match specified 'hdf5' type tag that requires 'signed char *'}}
#endif
  H5Dwrite(H5T_NATIVE_SCHAR, schar_buf); // no-warning
  H5Dwrite(H5T_NATIVE_UCHAR, uchar_buf); // no-warning
  H5Dwrite(H5T_NATIVE_INT,   int_buf);   // no-warning
  H5Dwrite(H5T_NATIVE_LONG,  long_buf);  // no-warning

#ifdef __CHAR_UNSIGNED__
  H5Dwrite(H5T_NATIVE_CHAR,  int_buf);  // expected-warning {{argument type 'int *' doesn't match specified 'hdf5' type tag that requires 'unsigned char *'}}
#else
  H5Dwrite(H5T_NATIVE_CHAR,  int_buf);  // expected-warning {{argument type 'int *' doesn't match specified 'hdf5' type tag that requires 'signed char *'}}
#endif
  H5Dwrite(H5T_NATIVE_INT,   long_buf); // expected-warning {{argument type 'long *' doesn't match specified 'hdf5' type tag that requires 'int *'}}

  // FIXME: we should warn here, but it will cause false positives because
  // different kinds may use same magic values.
  //H5Dwrite(MPI_INT, int_buf);
}

