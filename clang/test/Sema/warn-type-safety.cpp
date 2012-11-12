// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

typedef struct ompi_datatype_t *MPI_Datatype;

#define OMPI_PREDEFINED_GLOBAL(type, global) ((type) &(global))

#define MPI_FLOAT OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_float)
#define MPI_INT   OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_int)
#define MPI_NULL  OMPI_PREDEFINED_GLOBAL(MPI_Datatype, ompi_mpi_null)

extern struct ompi_predefined_datatype_t ompi_mpi_float __attribute__(( type_tag_for_datatype(mpi,float) ));
extern struct ompi_predefined_datatype_t ompi_mpi_int   __attribute__(( type_tag_for_datatype(mpi,int) ));
extern struct ompi_predefined_datatype_t ompi_mpi_null  __attribute__(( type_tag_for_datatype(mpi,void,must_be_null) ));

int f(int x) { return x; }
static const int wrong_init __attribute__(( type_tag_for_datatype(zzz,int) )) = f(100); // expected-error {{'type_tag_for_datatype' attribute requires the initializer to be an integral constant expression}}

//===--- Tests ------------------------------------------------------------===//
// Check that hidden 'this' is handled correctly.

class C
{
public:
  void f1(void *buf, int count, MPI_Datatype datatype)
       __attribute__(( pointer_with_type_tag(mpi,5,6) )); // expected-error {{attribute parameter 2 is out of bounds}}

  void f2(void *buf, int count, MPI_Datatype datatype)
       __attribute__(( pointer_with_type_tag(mpi,2,5) )); // expected-error {{attribute parameter 3 is out of bounds}}

  void f3(void *buf, int count, MPI_Datatype datatype)
       __attribute__(( pointer_with_type_tag(mpi,1,5) )); // expected-error {{attribute is invalid for the implicit this argument}}

  void f4(void *buf, int count, MPI_Datatype datatype)
       __attribute__(( pointer_with_type_tag(mpi,2,1) )); // expected-error {{attribute is invalid for the implicit this argument}}

  void MPI_Send(void *buf, int count, MPI_Datatype datatype)
       __attribute__(( pointer_with_type_tag(mpi,2,4) )); // no-error
};

// Check that we don't crash on type and value dependent expressions.
template<int a>
void value_dep(void *buf, int count, MPI_Datatype datatype)
     __attribute__(( pointer_with_type_tag(mpi,a,5) )); // expected-error {{attribute requires parameter 2 to be an integer constant}}

class OperatorIntStar
{
public:
  operator int*();
};

void test1(C *c, int *int_buf)
{
  c->MPI_Send(int_buf, 1, MPI_INT); // no-warning
  c->MPI_Send(int_buf, 1, MPI_FLOAT); // expected-warning {{argument type 'int *' doesn't match specified 'mpi' type tag that requires 'float *'}}
  c->MPI_Send(0, 0, MPI_INT); // no-warning
  c->MPI_Send(nullptr, 0, MPI_INT); // no-warning

  OperatorIntStar i;
  c->MPI_Send(i, 1, MPI_INT); // no-warning
  c->MPI_Send(i, 1, MPI_FLOAT); // expected-warning {{argument type 'int *' doesn't match specified 'mpi' type tag that requires 'float *'}}
}

template<typename T>
void test2(C *c, int *int_buf, T tag)
{
  c->MPI_Send(int_buf, 1, tag); // no-warning
}

void test3(C *c, int *int_buf) {
  test2(c, int_buf, MPI_INT);
  test2(c, int_buf, MPI_NULL);
}

