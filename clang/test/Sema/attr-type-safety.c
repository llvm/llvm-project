// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {};

typedef struct A *MPI_Datatype;

extern struct A datatype_wrong1 [[clang::type_tag_for_datatype]]; // expected-error {{'type_tag_for_datatype' attribute requires parameter 1 to be an identifier}}

extern struct A datatype_wrong2 [[clang::type_tag_for_datatype(mpi,1,2)]]; // expected-error {{expected a type}}

extern struct A datatype_wrong3 [[clang::type_tag_for_datatype(mpi,not_a_type)]]; // expected-error {{unknown type name 'not_a_type'}}

extern struct A datatype_wrong4 [[clang::type_tag_for_datatype(mpi,int,int)]]; // expected-error {{expected identifier}}

extern struct A datatype_wrong5 [[clang::type_tag_for_datatype(mpi,int,not_a_flag)]]; // expected-error {{invalid comparison flag 'not_a_flag'}}

extern struct A datatype_wrong6 [[clang::type_tag_for_datatype(mpi,int,layout_compatible,not_a_flag)]]; // expected-error {{invalid comparison flag 'not_a_flag'}}

extern struct A A_tag [[clang::type_tag_for_datatype(a,int)]];
extern struct A B_tag [[clang::type_tag_for_datatype(b,int)]];

static const int C_tag [[clang::type_tag_for_datatype(c,int)]] = 10;
static const int D_tag [[clang::type_tag_for_datatype(d,int)]] = 20;

[[clang::pointer_with_type_tag]] // expected-error {{'pointer_with_type_tag' attribute requires exactly 3 arguments}}
int wrong1(void *buf, MPI_Datatype datatype);

[[clang::pointer_with_type_tag(mpi,0,7)]]  // expected-error {{attribute parameter 2 is out of bounds}}
int wrong2(void *buf, MPI_Datatype datatype);

[[clang::pointer_with_type_tag(mpi,3,7)]] // expected-error {{attribute parameter 2 is out of bounds}}
int wrong3(void *buf, MPI_Datatype datatype);

[[clang::pointer_with_type_tag(mpi,1,0)]] // expected-error {{attribute parameter 3 is out of bounds}}
int wrong4(void *buf, MPI_Datatype datatype);

[[clang::pointer_with_type_tag(mpi,1,3)]] // expected-error {{attribute parameter 3 is out of bounds}}
int wrong5(void *buf, MPI_Datatype datatype);

[[clang::pointer_with_type_tag(mpi,0x8000000000000001ULL,1)]] // expected-error {{attribute parameter 2 is out of bounds}}
int wrong6(void *buf, MPI_Datatype datatype);

[[clang::pointer_with_type_tag(a,1,2)]] void A_func(void *ptr, void *tag);
[[clang::pointer_with_type_tag(c,1,2)]] void C_func(void *ptr, int tag);

