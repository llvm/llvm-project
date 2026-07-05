#include "timing.h"
#include <stdio.h>

#ifndef LIBNAME
#define LIBNAME UNKNOWN
#endif

#define LIBSTRING		LIBSTRINGX(LIBNAME)
#define LIBSTRINGX(a)	LIBSTRINGXX(a)
#define LIBSTRINGXX(a)	#a

double __floatundidf(uint64_t x);

int main(int argc, char *argv[]) {
#define INPUT_SIZE 512
	uint64_t input[INPUT_SIZE];
	int i, j;
	
	srand(42);
	
	// Initialize the input array with data of various sizes.
	for (i=0; i<INPUT_SIZE; ++i)
		input[i] = (((uint64_t)rand() << 32) | (uint64_t)rand()) >> (rand() & 63);
	
	double bestTime = __builtin_inf();
	void *dummyp;
	for (j=0; j<1024; ++j) {

		uint64_t startTime = mach_absolute_time();
		for (i=0; i<INPUT_SIZE; ++i)
			__floatundidf(input[i]);
		uint64_t endTime = mach_absolute_time();
		
		double thisTime = intervalInCycles(startTime, endTime);
		bestTime = __builtin_fmin(thisTime, bestTime);
		
		// Move the stack alignment between trials to eliminate (mostly) aliasing effects
		dummyp = alloca(1);
	}
	
	printf("%16s: %f cycles.\n", LIBSTRING, bestTime / (double) INPUT_SIZE);
	
	return 0;
}
