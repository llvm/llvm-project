!RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -fdebug-dump-symbols %s | FileCheck %s

program omp_default_category
  integer :: a
  integer :: b(10)

  !$omp parallel default(private:scalar)
    !CHECK: a (OmpPrivate): HostAssoc
    !CHECK: b (OmpShared): HostAssoc
    a = 1
    b = 1
  !$omp end parallel

end program omp_default_category
