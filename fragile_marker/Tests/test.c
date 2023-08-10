#include <stdio.h>
int f();
float add(int, float);
int main(){
    printf("%f\n", add(1, 2));
}
float add(int n1, float n2){
    return n1+n2;
}
void ff(){
    int a;
    int b;
    int c;
    int d;
    int e;
    int f;
}
int f(){
    int a=5, b=6, c=7, d=8, e;
    return a;
    e=a+b+c+d;
    return e;
}

char fff(double a, char *b, int *c){
    double res;
    res = a+(*b)/(*c);
    return (char)res;
}