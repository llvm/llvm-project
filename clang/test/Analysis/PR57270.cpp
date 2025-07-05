// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s

using size_t = __typeof(sizeof(int));

void clang_analyzer_explain(int);
void clang_analyzer_dump(int);
void *memset(void *, int, size_t);

struct S
{
    static int a;
    ~S(){};
};

int S::a = 0;

void foo()
{
    S::a = 0;

    int x = 3;
    memset(&x, 1, sizeof(x));

    S *arr = new S[x];
    delete[] arr;

    clang_analyzer_dump(S::a); // expected-warning-re{{{{derived_\$[0-9]+{conj_\$[0-9]+{int, LC[0-9]+, S[0-9]+, #[0-9]+},a}}}}}

    clang_analyzer_explain(S::a); // expected-warning-re{{{{value derived from \(symbol of type 'int' conjured at CFG element '->~S\(\) \(Implicit destructor\)'\) for global variable 'S::a'}}}}
}
