// RUN: %clang_cc1 -pedantic -fsyntax-only -verify %s

@protocol MyProto1 
@end

@protocol MyProto2 
@end

@interface INTF @end

id<MyProto1> Func(INTF <MyProto1, MyProto2> *p2) // expected-note 2{{passing argument to parameter 'p2' here}}
{
	return p2;
}




 id<MyProto1> Gunc(id <MyProto1, MyProto2>p2)
{
	return p2;
}


 id<MyProto1> Gunc1(id <MyProto1, MyProto2>p2)
{
	return p2;
}

id<MyProto1, MyProto2> Gunc2(id <MyProto1>p2)
{
	Func(p2);	// expected-warning {{passing 'id<MyProto1>' to parameter of incompatible type 'INTF<MyProto1,MyProto2> *'}}
	return p2;	// expected-warning {{returning 'id<MyProto1>' from a function with incompatible result type 'id<MyProto1,MyProto2>'}}
}



id<MyProto1> Gunc3(id <MyProto2>p2)
{
	return p2;	 // expected-warning {{returning 'id<MyProto2>' from a function with incompatible result type 'id<MyProto1>'}}
}


id<MyProto1, MyProto2> Gunc4(id <MyProto2, MyProto1>p2)
{
	return p2;
}



INTF<MyProto1> * Hunc(id <MyProto1, MyProto2>p2)
{
	return p2;
}


INTF<MyProto1> * Hunc1(id <MyProto1, MyProto2>p2)
{
	return p2;
}

INTF<MyProto1, MyProto2> * Hunc2(id <MyProto1>p2)
{
	Func(p2);	// expected-warning {{passing 'id<MyProto1>' to parameter of incompatible type 'INTF<MyProto1,MyProto2> *'}}
	return p2;	// expected-warning {{returning 'id<MyProto1>' from a function with incompatible result type 'INTF<MyProto1,MyProto2> *'}}
}

INTF<MyProto1> * Hunc3(id <MyProto2>p2)
{
	return p2;	 // expected-warning {{returning 'id<MyProto2>' from a function with incompatible result type 'INTF<MyProto1> *'}}
}


INTF<MyProto1, MyProto2> * Hunc4(id <MyProto2, MyProto1>p2)
{
	return p2;
}

id Iunc(id <MyProto1, MyProto2>p2)
{
	return p2;
}


id<MyProto1> Iunc1(id p2)
{
	return p2;
}

id<MyProto1, MyProto2> Iunc2(id p2)
{
	Iunc(p2);	
	return p2;
}
