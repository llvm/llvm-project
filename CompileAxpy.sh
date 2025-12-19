flang -O3 -fopenmp -fopenmp-version=60 omp-workdistribute.f90  -o a.out -mmlir --jit-workdistribute "$@"
