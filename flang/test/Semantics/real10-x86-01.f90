! RUN: %python %S/test_symbols.py %s %flang_fc1 -triple x86_64-unknown-linux-gnu

 !DEF: /MainProgram1/rpdt DerivedType
 !DEF: /MainProgram1/rpdt/k TypeParam INTEGER(4)
 type :: rpdt(k)
  !REF: /MainProgram1/rpdt/k
  integer, kind :: k
  !REF: /MainProgram1/rpdt/k
  !DEF: /MainProgram1/rpdt/x ObjectEntity REAL(int(int(k,kind=4),kind=8))
  real(kind=k) :: x
 end type rpdt
 !DEF: /MainProgram1/zpdt DerivedType
 !DEF: /MainProgram1/zpdt/k TypeParam INTEGER(4)
 type :: zpdt(k)
  !REF: /MainProgram1/zpdt/k
  integer, kind :: k
  !REF: /MainProgram1/zpdt/k
  !DEF: /MainProgram1/zpdt/x ObjectEntity COMPLEX(int(int(k,kind=4),kind=8))
  complex(kind=k) :: x
 end type zpdt
 !REF: /MainProgram1/rpdt
 !DEF: /MainProgram1/a10 ObjectEntity TYPE(rpdt(k=10_4))
 type(rpdt(10)) :: a10
 !REF: /MainProgram1/zpdt
 !DEF: /MainProgram1/z10 ObjectEntity TYPE(zpdt(k=10_4))
 type(zpdt(10)) :: z10
end program
