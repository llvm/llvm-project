struct foo
{
    int a;
    int b;
    int c;
    int d;
    int e;
    int f;
    int g;
    int h;
    int i;
    int j;
    int k;
    int l;
    int m;
    int n;
    int o;
    int p;
    int q;
    int r;
    
    foo(int X) :
    a(X),
    b(X+1),
    c(X+3),
    d(X+5),
    e(X+7),
    f(X+9),
    g(X+11),
    h(X+13),
    i(X+15),
    j(X+17),
    k(X+19),
    l(X+21),
    m(X+23),
    n(X+25),
    o(X+27),
    p(X+29),
    q(X+31),
    r(X+33) {}
};

struct wrapint
{
    int x;
    wrapint(int X) : x(X) {}
};

int main()
{
    foo f00_1(1);
    foo *f00_ptr = new foo(12);
    
    f00_1.a++; // Set break point at this line.
    
    wrapint test_cast('A' +
               256*'B' +
               256*256*'C'+
               256*256*256*'D');
    // Set cast break point at this line.
    test_cast.x = 'Q' +
	               256*'X' +
	               256*256*'T'+
	               256*256*256*'F';
    return 0;     // Set second cast break point at this line.
}
