! RUN: %python %S/test_modfile.py %s %flang_fc1
MODULE M1
    TYPE T1
        REAL(KIND=4), DIMENSION(:, :), POINTER    :: ptr => Null()
    END TYPE T1

    TYPE O1
        TYPE(T1), POINTER           :: d => Null()
    END TYPE O1
END MODULE

!Expect: m1.mod
!module m1
!type::t1
!real(4),pointer::ptr(:,:)=>NULL()
!end type
!intrinsic::null
!type::o1
!type(t1),pointer::d=>NULL()
!end type
!end

MODULE M2
    USE M1,only : &
    o1_prv => o1

    public
    TYPE D1
        TYPE(o1_prv), PRIVATE        :: prv = o1_prv ()
    END TYPE D1
END MODULE

!Expect: m2.mod
!module m2
!use m1,only:o1_prv=>o1
!type::d1
!type(o1_prv),private::prv=o1_prv(d=NULL())
!end type
!end

MODULE M3
    USE M2 , only : d1_prv => D1

    PUBLIC
    TYPE d1_ext
        TYPE(d1_prv), PRIVATE :: prv = d1_prv()
    END TYPE
END MODULE

!Expect: m3.mod
!module m3
!use m2,only:o1_prv
!use m2,only:d1_prv=>d1
!private::o1_prv
!type::d1_ext
!type(d1_prv),private::prv=d1_prv(prv=o1_prv(d=NULL()))
!end type
!end
