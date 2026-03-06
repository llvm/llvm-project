// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s

// Compound literal doesn't need a constant expression inside a initializer-list if it is already inside a function 
// see: https://github.com/llvm/llvm-project/issues/87867
int foo(int *a, int b) {
    return 0;
}

int x;
struct{int t;} a = (struct {
    typeof(foo(&(struct { int t; }){.t = x}.t, 0)) t; // expected-error {{initializer element is not a compile-time constant}}
}){0};

void inside_a_func(){
    int x;
    (void)(struct {
        typeof(foo(&(struct { int t; }){.t = x}.t, 0)) t;
    }){0};
}

// see: https://github.com/llvm/llvm-project/issues/143613
#define bitcast(type, value) \
    (((union{ typeof(value) src; type dst; }){ (value) }).dst)

double placeholder = 10.0;
double bar = bitcast(double, placeholder);  // expected-error {{initializer element is not a compile-time constant}}

int main(void)
{
    int foo = 4;
    foo = bitcast(int, bitcast(double, foo));
    return 0;
}
