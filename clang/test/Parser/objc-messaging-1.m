// RUN: %clang_cc1 %s -fsyntax-only -verify
int main ()
{
	int i,j;
	struct S *p;
        id a, b, c;
	[a ii]; // expected-warning{{not found}}
	[a if: 1 :2]; // expected-warning{{not found}}
	[a inout: 1 :2 another:(2,3,4)]; // expected-warning{{not found}} \
           // expected-warning 2{{left operand of comma operator has no effect}}
	[a inout: 1 :2 another:(2,3,4), 6,6,8]; // expected-warning{{not found}} \
           // expected-warning 2{{left operand of comma operator has no effect}}
	[a inout: 1 :2 another:(2,3,4), (6,4,5),6,8]; // expected-warning{{not found}} \
           // expected-warning 4{{left operand of comma operator has no effect}}
	[a inout: 1 :2 another:(i+10), (i,j-1,5),6,8]; // expected-warning{{not found}} \
           // expected-warning 2{{left operand of comma operator has no effect}}
	[a long: 1 :2 another:(i+10), (i,j-1,5),6,8]; // expected-warning{{not found}} \
           // expected-warning 2{{left operand of comma operator has no effect}}
	[a : "Hello\n" :2 another:(i+10), (i,j-1,5),6,8]; // expected-warning{{not found}} \
           // expected-warning 2{{left operand of comma operator has no effect}}

	// Comma expression as receiver (rdar://6222856)
	[a, b, c foo]; // expected-warning{{not found}} \
           // expected-warning 2{{left operand of comma operator has no effect}}

}
