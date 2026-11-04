// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

void clang_analyzer_eval(bool);

struct S{
    static int CtorInvocationCount;
    static int DtorInvocationCount;

    S(){CtorInvocationCount++;}
    ~S(){DtorInvocationCount++;}
};

int S::CtorInvocationCount = 0;
int S::DtorInvocationCount = 0;

void zeroSizeArrayStack() {
    S::CtorInvocationCount = 0;

    S arr[0];

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
}

void zeroSizeMultidimensionalArrayStack() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;

    {
        S arr[2][0];
        S arr2[0][2];

        S arr3[0][2][2];
        S arr4[2][2][0];
        S arr5[2][0][2];
    }

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}

void zeroSizeArrayStackInLambda() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;

    []{
        S arr[0];
    }();    
    
    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}

void zeroSizeArrayHeap() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;

    auto *arr = new S[0];
    delete[] arr;
    
    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}

void zeroSizeMultidimensionalArrayHeap() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;

    auto *arr = new S[2][0];
    delete[] arr;
        
    auto *arr2 = new S[0][2];
    delete[] arr2;

    auto *arr3 = new S[0][2][2];
    delete[] arr3;

    auto *arr4 = new S[2][2][0];
    delete[] arr4;

    auto *arr5 = new S[2][0][2];
    delete[] arr5;

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}

#if __cplusplus >= 201703L

void zeroSizeArrayBinding() {
    S::CtorInvocationCount = 0;

    S arr[0];

    // Note: This is an error in gcc but a warning in clang.
    // In MSVC the declaration of 'S arr[0]' is already an error
    // and it doesn't recognize this syntax as a structured binding.
    auto [] = arr; //expected-warning{{ISO C++17 does not allow a structured binding group to be empty}}

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
}

#endif

void zeroSizeArrayLambdaCapture() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;
        
    S arr[0];

    auto l = [arr]{};
    [arr]{}();    
    
    //FIXME: These should be TRUE. We should avoid calling the destructor 
    // of the temporary that is materialized as the lambda.
    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}} expected-warning{{FALSE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}} expected-warning{{FALSE}}
}

// FIXME: Report a warning if the standard is at least C++17.
#if __cplusplus < 201703L
void zeroSizeArrayLambdaCaptureUndefined1() {
    S arr[0];
    int n;

    auto l = [arr, n]{
        int x = n; //expected-warning{{Assigned value is uninitialized}}
        (void) x;
    };

    l();
}
#endif

void zeroSizeArrayLambdaCaptureUndefined2() {
    S arr[0];
    int n;

    [arr, n]{
        int x = n; //expected-warning{{Assigned value is uninitialized}}
        (void) x;
    }();
}

struct Wrapper{
    S arr[0];
};

void zeroSizeArrayMember() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;

    {
        Wrapper W;
    }

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}

void zeroSizeArrayMemberCopyMove() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;
    
    {
        Wrapper W;
        Wrapper W2 = W;
        Wrapper W3 = (Wrapper&&) W2;
    }

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}

struct MultiWrapper{
    S arr[2][0];
};

void zeroSizeMultidimensionalArrayMember() {
    S::CtorInvocationCount = 0;
    S::DtorInvocationCount = 0;

    {
        MultiWrapper MW;
    }

    clang_analyzer_eval(S::CtorInvocationCount == 0); //expected-warning{{TRUE}}
    clang_analyzer_eval(S::DtorInvocationCount == 0); //expected-warning{{TRUE}}
}
