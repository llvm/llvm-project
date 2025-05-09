// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=60 -fsyntax-only -Wuninitialized -verify %s

void func() {

    // expected-error@+2 {{statement after '#pragma omp fuse' must be a loop sequence containing canonical loops or loop-generating constructs}}
    #pragma omp fuse 
    ;

    // expected-error@+2 {{statement after '#pragma omp fuse' must be a for loop}}
    #pragma omp fuse 
    {int bar = 0;}

    // expected-error@+4 {{statement after '#pragma omp fuse' must be a for loop}}
    #pragma omp fuse 
    {
        for(int i = 0; i < 10; ++i);
        int x = 2;
    }

    // expected-error@+2 {{statement after '#pragma omp fuse' must be a loop sequence containing canonical loops or loop-generating constructs}}
    #pragma omp fuse 
    #pragma omp for 
    for (int i = 0; i < 7; ++i)
        ;

    {
        // expected-error@+2 {{expected statement}}
        #pragma omp fuse
    }

    // expected-warning@+1 {{extra tokens at the end of '#pragma omp fuse' are ignored}}
    #pragma omp fuse foo
    {
        for (int i = 0; i < 7; ++i)
            ;
    }


    // expected-error@+1 {{unexpected OpenMP clause 'final' in directive '#pragma omp fuse'}}
    #pragma omp fuse final(0) 
    {
        for (int i = 0; i < 7; ++i)
            ;
    }

    //expected-error@+4 {{loop after '#pragma omp fuse' is not in canonical form}}
    //expected-error@+3 {{increment clause of OpenMP for loop must perform simple addition or subtraction on loop variable 'i'}}
    #pragma omp fuse 
    {
        for(int i = 0; i < 10; i*=2) {
            ;
        }
    }

    //expected-error@+2 {{loop sequence after '#pragma omp fuse' must contain at least 1 canonical loop or loop-generating construct}}
    #pragma omp fuse 
    {}

    //expected-error@+3 {{statement after '#pragma omp fuse' must be a for loop}}
    #pragma omp fuse 
    {
        #pragma omp unroll full 
        for(int i = 0; i < 10; ++i);
        
        for(int j = 0; j < 10; ++j);
    }

    //expected-warning@+5 {{loop sequence following '#pragma omp fuse' contains induction variables of differing types: 'int' and 'unsigned int'}}
    //expected-warning@+5 {{loop sequence following '#pragma omp fuse' contains induction variables of differing types: 'int' and 'long long'}}
    #pragma omp fuse 
    {
        for(int i = 0; i < 10; ++i);
        for(unsigned int j = 0; j < 10; ++j);
        for(long long k = 0; k < 100; ++k);
    }
}