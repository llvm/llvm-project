// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -std=c++20 -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing 
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -std=c++20 -fopenmp-version=60 -ast-dump  %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -std=c++20 -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT

// Check same results after serialization round-trip 
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -std=c++20 -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -std=c++20 -fopenmp-version=60 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -std=c++20 -fopenmp-version=60 -include-pch %t -ast-print    %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER 

// placeholder for loop body code
extern "C" void body(...);

// PRINT-LABEL: void foo1(
// DUMP-LABEL: FunctionDecl {{.*}} foo1
void foo1() {
    // PRINT: #pragma omp fuse
    // DUMP:  OMPFuseDirective
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: for (int i = 0; i < 10; i += 2)
        // DUMP: ForStmt
        for (int i = 0; i < 10; i += 2)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);
        // PRINT: for (int j = 10; j > 0; --j)
        // DUMP: ForStmt
        for (int j = 10; j > 0; --j)
            // PRINT: body(j)
            // DUMP: CallExpr
            body(j);
        // PRINT: for (int k = 0; k <= 10; ++k)
        // DUMP: ForStmt
        for (int k = 0; k <= 10; ++k)
            // PRINT: body(k)
            // DUMP: CallExpr
            body(k);

    }

}

// PRINT-LABEL: void foo2(
// DUMP-LABEL: FunctionDecl {{.*}} foo2
void foo2() {
    // PRINT: #pragma omp unroll partial(4)
    // DUMP: OMPUnrollDirective
    // DUMP-NEXT: OMPPartialClause
    // DUMP-NEXT: ConstantExpr
    // DUMP-NEXT: value: Int 4
    // DUMP-NEXT: IntegerLiteral {{.*}} 4
    #pragma omp unroll partial(4)
    // PRINT: #pragma omp fuse
    // DUMP-NEXT: OMPFuseDirective 
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: for (int i = 0; i < 10; i += 2)
        // DUMP: ForStmt
        for (int i = 0; i < 10; i += 2)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);
        // PRINT: for (int j = 10; j > 0; --j)
        // DUMP: ForStmt
        for (int j = 10; j > 0; --j)
            // PRINT: body(j)
            // DUMP: CallExpr
            body(j);  
    }    
    
}

//PRINT-LABEL: void foo3(
//DUMP-LABEL: FunctionTemplateDecl {{.*}} foo3
template<int Factor1, int Factor2> 
void foo3() {
    // PRINT:  #pragma omp fuse
    // DUMP: OMPFuseDirective
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: #pragma omp unroll partial(Factor1)
        // DUMP: OMPUnrollDirective
        #pragma omp unroll partial(Factor1)
        // PRINT: for (int i = 0; i < 12; i += 1)
        // DUMP: ForStmt
        for (int i = 0; i < 12; i += 1)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);
        // PRINT: #pragma omp unroll partial(Factor2)
        // DUMP: OMPUnrollDirective
        #pragma omp unroll partial(Factor2)
        // PRINT: for (int k = 0; k <= 10; ++k)
        // DUMP: ForStmt
        for (int k = 0; k <= 10; ++k)
            // PRINT: body(k)
            // DUMP: CallExpr
            body(k);

    }
}

// Also test instantiating the template.
void tfoo3() {
    foo3<4,2>();
}

//PRINT-LABEL: void foo4(
//DUMP-LABEL: FunctionTemplateDecl {{.*}} foo4
template<typename T, T Step> 
void foo4(int start, int end) {
    // PRINT:  #pragma omp fuse
    // DUMP: OMPFuseDirective
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: for (T i = start; i < end; i += Step)
        // DUMP: ForStmt
        for (T i = start; i < end; i += Step)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);

        // PRINT: for (T j = end; j > start; j -= Step)
        // DUMP: ForStmt 
        for (T j = end; j > start; j -= Step) {
            // PRINT: body(j)
            // DUMP: CallExpr
            body(j);
        }

    }
}

// Also test instantiating the template.
void tfoo4() {
    foo4<int, 4>(0, 64);
}



// PRINT-LABEL: void foo5(
// DUMP-LABEL: FunctionDecl {{.*}} foo5
void foo5() {
    double arr[128], arr2[128];
    // PRINT: #pragma omp fuse
    // DUMP:  OMPFuseDirective
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT-NEXT: for (auto &&a : arr)
        // DUMP-NEXT: CXXForRangeStmt
        for (auto &&a: arr)
            // PRINT: body(a)
            // DUMP: CallExpr
            body(a);
        // PRINT: for (double v = 42; auto &&b : arr)
        // DUMP: CXXForRangeStmt
        for (double v = 42; auto &&b: arr)
            // PRINT: body(b, v);
            // DUMP: CallExpr
            body(b, v);
        // PRINT: for (auto &&c : arr2)
        // DUMP: CXXForRangeStmt
        for (auto &&c: arr2)
            // PRINT: body(c)
            // DUMP: CallExpr
            body(c);

    }

}

// PRINT-LABEL: void foo6(
// DUMP-LABEL: FunctionDecl {{.*}} foo6
void foo6() {
    // PRINT: #pragma omp fuse
    // DUMP: OMPFuseDirective
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt
    {
        // PRINT: #pragma omp fuse
        // DUMP: OMPFuseDirective
        #pragma omp fuse 
        // PRINT: {
        // DUMP: CompoundStmt
        {
            // PRINT: for (int i = 0; i <= 10; ++i)
            // DUMP: ForStmt
            for (int i = 0; i <= 10; ++i)
                body(i);
            // PRINT: for (int j = 0; j < 100; ++j)
            // DUMP: ForStmt
            for(int j = 0; j < 100; ++j)
                body(j);
        }
        // PRINT: #pragma omp unroll partial(4)
        // DUMP: OMPUnrollDirective
        #pragma omp unroll partial(4)
        // PRINT: for (int k = 0; k < 250; ++k)
        // DUMP: ForStmt
        for (int k = 0; k < 250; ++k) 
            body(k);
    }
}

// PRINT-LABEL: void foo7(
// DUMP-LABEL: FunctionDecl {{.*}} foo7
void foo7() {
    // PRINT: #pragma omp fuse
    // DUMP:  OMPFuseDirective
    #pragma omp fuse 
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: {
        // DUMP: CompoundStmt   
        {
            // PRINT: {
            // DUMP: CompoundStmt   
            {
                // PRINT: for (int i = 0; i < 10; i += 2)
                // DUMP: ForStmt
                for (int i = 0; i < 10; i += 2)
                    // PRINT: body(i)
                    // DUMP: CallExpr
                    body(i);
                // PRINT: for (int j = 10; j > 0; --j)
                // DUMP: ForStmt
                for (int j = 10; j > 0; --j)
                    // PRINT: body(j)
                    // DUMP: CallExpr
                    body(j);
            }
        }
        // PRINT: {
        // DUMP: CompoundStmt   
        {
            // PRINT: {
            // DUMP: CompoundStmt   
            {
                // PRINT: {
                // DUMP: CompoundStmt   
                {
                    // PRINT: for (int k = 0; k <= 10; ++k)
                    // DUMP: ForStmt
                    for (int k = 0; k <= 10; ++k)
                        // PRINT: body(k)
                        // DUMP: CallExpr
                        body(k);
                }
            }
        }
    }

}

// PRINT-LABEL: void foo8(
// DUMP-LABEL: FunctionDecl {{.*}} foo8
void foo8() {
    // PRINT: #pragma omp fuse looprange(2,2)
    // DUMP:  OMPFuseDirective
    // DUMP: OMPLooprangeClause
    #pragma omp fuse looprange(2,2)
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: for (int i = 0; i < 10; i += 2)
        // DUMP: ForStmt
        for (int i = 0; i < 10; i += 2)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);
        // PRINT: for (int j = 10; j > 0; --j)
        // DUMP: ForStmt
        for (int j = 10; j > 0; --j)
            // PRINT: body(j)
            // DUMP: CallExpr
            body(j);
        // PRINT: for (int k = 0; k <= 10; ++k)
        // DUMP: ForStmt
        for (int k = 0; k <= 10; ++k)
            // PRINT: body(k)
            // DUMP: CallExpr
            body(k);

    }

}

//PRINT-LABEL: void foo9(
//DUMP-LABEL: FunctionTemplateDecl {{.*}} foo9
//DUMP-LABEL: NonTypeTemplateParmDecl {{.*}} F
//DUMP-LABEL: NonTypeTemplateParmDecl {{.*}} C
template<int F, int C> 
void foo9() {
    // PRINT:  #pragma omp fuse looprange(F,C)
    // DUMP: OMPFuseDirective
    // DUMP: OMPLooprangeClause
    #pragma omp fuse looprange(F,C)
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: for (int i = 0; i < 10; i += 2)
        // DUMP: ForStmt
        for (int i = 0; i < 10; i += 2)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);
        // PRINT: for (int j = 10; j > 0; --j)
        // DUMP: ForStmt
        for (int j = 10; j > 0; --j)
            // PRINT: body(j)
            // DUMP: CallExpr
            body(j);

    }
}

// Also test instantiating the template.
void tfoo9() {
    foo9<1, 2>();
}

// PRINT-LABEL: void foo10(
// DUMP-LABEL: FunctionDecl {{.*}} foo10
void foo10() {
    // PRINT: #pragma omp fuse looprange(2,2)
    // DUMP:  OMPFuseDirective
    // DUMP: OMPLooprangeClause
    #pragma omp fuse looprange(2,2)
    // PRINT: {
    // DUMP: CompoundStmt       
    {
        // PRINT: for (int i = 0; i < 10; i += 2)
        // DUMP: ForStmt
        for (int i = 0; i < 10; i += 2)
            // PRINT: body(i)
            // DUMP: CallExpr
            body(i);
        // PRINT: for (int ii = 0; ii < 10; ii += 2)
        // DUMP: ForStmt
        for (int ii = 0; ii < 10; ii += 2)
            // PRINT: body(ii)
            // DUMP: CallExpr
            body(ii);
        // PRINT: #pragma omp fuse looprange(2,2)
        // DUMP:  OMPFuseDirective
        // DUMP: OMPLooprangeClause
        #pragma omp fuse looprange(2,2)
        {
            // PRINT: for (int j = 10; j > 0; --j)
            // DUMP: ForStmt
            for (int j = 10; j > 0; --j)
                // PRINT: body(j)
                // DUMP: CallExpr
                body(j);
            // PRINT: for (int jj = 10; jj > 0; --jj)
            // DUMP: ForStmt
            for (int jj = 10; jj > 0; --jj)
                // PRINT: body(jj)
                // DUMP: CallExpr
                body(jj);
            // PRINT: for (int k = 0; k <= 10; ++k)
            // DUMP: ForStmt
            for (int k = 0; k <= 10; ++k)
                // PRINT: body(k)
                // DUMP: CallExpr
                body(k);
            // PRINT: for (int kk = 0; kk <= 10; ++kk)
            // DUMP: ForStmt
            for (int kk = 0; kk <= 10; ++kk)
                // PRINT: body(kk)
                // DUMP: CallExpr
                body(kk);
        }
    }

}

#endif
