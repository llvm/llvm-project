// RUN: %clang_analyze_cc1 -std=c11 -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_dump(int*);

const int const_index = 1;
extern int unknown_index;
extern int array[3];
extern int matrix[3][3];

int main(){

    // expected-warning@+1 {{&Element{array,1 S64b,int}}}
    clang_analyzer_dump(&array[const_index]);

    // expected-warning@+1 {{&Element{array,reg_$1<int unknown_index>,int}}}
    clang_analyzer_dump(&array[unknown_index]);
    
    // expected-warning@+1 {{&Element{Element{matrix,1 S64b,int[3]},1 S64b,int}}}
    clang_analyzer_dump(&matrix[const_index][const_index]);

    // expected-warning@+1 {{&Element{Element{matrix,reg_$1<int unknown_index>,int[3]},1 S64b,int}}}
    clang_analyzer_dump(&matrix[unknown_index][const_index]);

    // expected-warning@+1 {{&Element{Element{matrix,1 S64b,int[3]},reg_$1<int unknown_index>,int}}}
    clang_analyzer_dump(&matrix[const_index][unknown_index]);

    // expected-warning@+1 {{&Element{Element{matrix,reg_$1<int unknown_index>,int[3]},reg_$1<int unknown_index>,int}}}
    clang_analyzer_dump(&matrix[unknown_index][unknown_index]);

    return 0;
}
