! RUN: %flang_fc1 -fopenmp -fdebug-dump-symbols %s | FileCheck %s

! Test symbols generated in block constructs in the 
! presence of `default(...)` clause

program sample
    !CHECK: a size=4 offset=20: ObjectEntity type: INTEGER(4)
    !CHECK: k size=4 offset=16: ObjectEntity type: INTEGER(4)
    !CHECK: w size=4 offset=12: ObjectEntity type: INTEGER(4)
    !CHECK: x size=4 offset=0: ObjectEntity type: INTEGER(4)
    !CHECK: y size=4 offset=4: ObjectEntity type: INTEGER(4)
    !CHECK: z size=4 offset=8: ObjectEntity type: INTEGER(4)
    integer x, y, z, w, k, a 
    !$omp parallel  firstprivate(x) private(y) shared(w) default(private)
        !CHECK: OtherConstruct scope: size=0 alignment=1
        !CHECK: a (OmpPrivate): HostAssoc
        !CHECK: k (OmpPrivate): HostAssoc
        !CHECK: x (OmpFirstPrivate): HostAssoc
        !CHECK: y (OmpPrivate): HostAssoc
        !CHECK: z (OmpPrivate): HostAssoc
        !$omp parallel default(private)
            !CHECK: OtherConstruct scope: size=0 alignment=1
            !CHECK: a (OmpPrivate): HostAssoc
            !CHECK: x (OmpPrivate): HostAssoc
            !CHECK: y (OmpPrivate): HostAssoc
            y = 20
            x = 10
           !$omp parallel
                !CHECK: OtherConstruct scope: size=0 alignment=1
                a = 10
           !$omp end parallel
        !$omp end parallel 

        !$omp parallel default(firstprivate) shared(y) private(w)
            !CHECK: OtherConstruct scope: size=0 alignment=1
            !CHECK: k (OmpFirstPrivate): HostAssoc
            !CHECK: w (OmpPrivate): HostAssoc
            !CHECK: z (OmpFirstPrivate): HostAssoc
            y = 30
            w = 40 
            z = 50 
            k = 40
        !$omp end parallel
    !$omp end parallel  
end program sample
