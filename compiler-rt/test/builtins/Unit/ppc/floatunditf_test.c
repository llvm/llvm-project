// REQUIRES: target-is-powerpc64le
// RUN: %clang_builtins %s %librt -o %t && %run %t
#include <stdint.h>
#include <stdio.h>
#include "int_lib.h"

COMPILER_RT_ABI long double __floatunditf(uint64_t);

#include "floatunditf_test.h"
#include "DD.h"

int main(int argc, char *argv[]) {
	int i;
	
	DD expected;
	DD computed;
	
	for (i=0; i<numTests; ++i) {
		expected.hi = tests[i].hi;
		expected.lo = tests[i].lo;
		computed.ld = __floatunditf(tests[i].input);
		
		if ((computed.hi != expected.hi) || (computed.lo != expected.lo))
		{
			printf("Error on __floatunditf( 0x%016llx ):\n", tests[i].input);
			printf("\tExpected %La = ( %a , %a )\n", expected.ld, expected.hi, expected.lo);
			printf("\tComputed %La = ( %a , %a )\n", computed.ld, computed.hi, computed.lo);
			return 1;
		}
	}
	
	return 0;
}
