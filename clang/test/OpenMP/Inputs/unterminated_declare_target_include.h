// expected-warning@+1 {{expected '#pragma omp end declare target' at end of file to match '#pragma omp declare target'}}
#pragma omp declare target
void zxy();
