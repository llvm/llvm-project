#include <iostream>
#include <string>

struct baz
    {
        int h;
        int k;
        baz(int a, int b) : h(a), k(b) {}
    };

struct bar
	{
		int i;
		int* i_ptr;
        baz b;
        baz& b_ref;
		bar(int x) : i(x),i_ptr(new int(x+1)),b(i+3,i+5),b_ref(b) {}
	};
	
struct foo
	{
		int a;
		int* a_ptr;
		bar b;
		
		foo(int x) : a(x),
		a_ptr(new int(x+1)),
		b(2*x) {}
		
	};
	
int main(int argc, char** argv)
{
	foo foo1(12);
	foo foo2(121);
	
	foo2.a = 7777; // Stop here
	*(foo2.b.i_ptr) = 8888;
    foo2.b.b.h = 9999;
	
	*(foo1.a_ptr) = 9999;
	foo1.b.i = 9999;
	
	int numbers[5] = {1,2,3,4,5};
	
	return 0;
	
}
