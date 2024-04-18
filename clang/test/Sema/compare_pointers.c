// RUN: %clang_cc1 -Wno-pointer-integer-compare -Wpointer-integer-ordered-compare -fsyntax-only -verify=pointer-integer-ordered %s
// RUN: %clang_cc1 -Wpointer-integer-compare -Wno-pointer-integer-ordered-compare -fsyntax-only -verify=pointer-integer %s

void test1(int *a){
    int b = 1;
    short c = 1;
    if(c<a) {}; // pointer-integer-ordered-warning{{ordered comparison between pointer and integer ('short' and 'int *')}}
    if(a!=b) {}; // pointer-integer-warning{{comparison between pointer and integer ('int *' and 'int')}}
    if(a == b) {}; // pointer-integer-warning{{comparison between pointer and integer ('int *' and 'int')}}
}

int test2(int *a){
    return a>=0; // pointer-integer-ordered-warning{{ordered comparison between pointer and zero ('int *' and 'int') is an extension}}
}

int test3(int *a){
    return a>=1; // pointer-integer-ordered-warning{{ordered comparison between pointer and integer ('int *' and 'int')}}
}

int test4(int *a){
    return a>1; // pointer-integer-ordered-warning{{ordered comparison between pointer and integer ('int *' and 'int')}}
}

int test5(int *a){
    int zero = 0;
    return a>=zero; // pointer-integer-ordered-warning{{ordered comparison between pointer and integer ('int *' and 'int')}}
}
