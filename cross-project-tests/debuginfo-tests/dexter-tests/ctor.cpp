// REQUIRES: lldb
// UNSUPPORTED: system-windows
// XFAIL: system-darwin
//
// RUN: %clang -std=gnu++11 -O0 -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t --debugger 'lldb' -- %s

class A {
public:
	A() : zero(0), data(42) { // DexLabel('ctor_start')
	}
private:
	int zero;
	int data;
};

int main() {
	A a;
	return 0;
}


/*
DexExpectProgramState({
	'frames': [
		{
			'location': {
				'lineno': ref('ctor_start')
			},
			'watches': {
				'*this': {'is_irretrievable': False}
			}
		}
	]
})
*/

