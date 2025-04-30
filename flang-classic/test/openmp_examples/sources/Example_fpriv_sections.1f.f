! @@name:	fpriv_sections.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program section
    use omp_lib
    integer :: section_count = 0
    integer, parameter :: NT = 4
    call omp_set_dynamic(.false.)
    call omp_set_num_threads(NT)
!$omp parallel
!$omp sections firstprivate ( section_count )
!$omp section
    section_count = section_count + 1
! may print the number one or two
    print *, 'section_count', section_count
!$omp section
    section_count = section_count + 1
! may print the number one or two
    print *, 'section_count', section_count
!$omp end sections
!$omp end parallel
end program section
