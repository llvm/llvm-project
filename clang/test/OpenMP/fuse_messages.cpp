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
        for(int j = 0; j < 100; ++j);

    }


    // expected-error@+1 {{unexpected OpenMP clause 'final' in directive '#pragma omp fuse'}}
    #pragma omp fuse final(0)
    {
        for (int i = 0; i < 7; ++i)
            ;
        for(int j = 0; j < 100; ++j);

    }

    //expected-error@+3 {{increment clause of OpenMP for loop must perform simple addition or subtraction on loop variable 'i'}}
    #pragma omp fuse
    {
        for(int i = 0; i < 10; i*=2) {
            ;
        }
        for(int j = 0; j < 100; ++j);
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

    //expected-warning@+2 {{looprange clause selects a single loop, resulting in redundant fusion}}
    #pragma omp fuse
    {
        for(int i = 0; i < 10; ++i);
    }

    //expected-warning@+1 {{looprange clause selects a single loop, resulting in redundant fusion}}
    #pragma omp fuse looprange(1, 1)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
    }

    //expected-error@+1 {{argument to 'looprange' clause must be a strictly positive integer value}}
    #pragma omp fuse looprange(1, -1)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
    }

    //expected-error@+1 {{argument to 'looprange' clause must be a strictly positive integer value}}
    #pragma omp fuse looprange(1, 0)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
    }

    const int x = 1;
    constexpr int y = 4;
    //expected-error@+1 {{looprange clause selects loops from 1 to 4 but this exceeds the number of loops (3) in the loop sequence}}
    #pragma omp fuse looprange(x,y)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
    }

    //expected-error@+1 {{looprange clause selects loops from 1 to 420 but this exceeds the number of loops (3) in the loop sequence}}
    #pragma omp fuse looprange(1,420)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
    }

    //expected-error@+1 {{looprange clause selects loops from 1 to 6 but this exceeds the number of loops (5) in the loop sequence}}
    #pragma omp fuse looprange(1,6)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
        // This fusion results in 2 loops
        #pragma omp fuse looprange(1,2)
        {
            for(int i = 0; i < 10; ++i);
            for(int j = 0; j < 100; ++j);
            for(int k = 0; k < 50; ++k);
        }
    }

    //expected-error@+1 {{looprange clause selects loops from 2 to 4 but this exceeds the number of loops (3) in the loop sequence}}
    #pragma omp fuse looprange(2,3)
    {
        #pragma omp unroll partial(2)
        for(int i = 0; i < 10; ++i);

        #pragma omp reverse
        for(int j = 0; j < 10; ++j);

        #pragma omp fuse
        {
            {
                #pragma omp reverse
                for(int j = 0; j < 10; ++j);
            }
            for(int k = 0; k < 50; ++k);
        }
    }
}

// In a template context, but expression itself not instantiation-dependent
template <typename T>
static void templated_func() {

    //expected-warning@+1 {{looprange clause selects a single loop, resulting in redundant fusion}}
    #pragma omp fuse looprange(2,1)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
    }

    //expected-error@+1 {{looprange clause selects loops from 3 to 5 but this exceeds the number of loops (3) in the loop sequence}}
    #pragma omp fuse looprange(3,3)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
    }

}

template <int V>
static void templated_func_value_dependent() {

    //expected-warning@+1 {{looprange clause selects a single loop, resulting in redundant fusion}}
    #pragma omp fuse looprange(V,1)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
    }
}

template <typename T>
static void templated_func_type_dependent() {
    constexpr T s = 1;

    //expected-error@+1 {{argument to 'looprange' clause must be a strictly positive integer value}}
    #pragma omp fuse looprange(s,s-1)
    {
        for(int i = 0; i < 10; ++i);
        for(int j = 0; j < 100; ++j);
        for(int k = 0; k < 50; ++k);
    }
}


void template_inst() {
    // expected-note@+1 {{in instantiation of function template specialization 'templated_func<int>' requested here}}
    templated_func<int>();
    // expected-note@+1 {{in instantiation of function template specialization 'templated_func_value_dependent<1>' requested here}}
    templated_func_value_dependent<1>();
    // expected-note@+1 {{in instantiation of function template specialization 'templated_func_type_dependent<int>' requested here}}
    templated_func_type_dependent<int>();
}


