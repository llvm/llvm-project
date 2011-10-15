#include "timing.h"
#include <stdio.h>

#define INPUT_TYPE uint64_t
#define INPUT_SIZE 512
#define FUNCTION_NAME __floatundisf

#ifndef LIBNAME
#define LIBNAME UNKNOWN
#endif

#define LIBSTRING		LIBSTRINGX(LIBNAME)
#define LIBSTRINGX(a)	LIBSTRINGXX(a)
#define LIBSTRINGXX(a)	#a

float FUNCTION_NAME(INPUT_TYPE x);

int main(int argc, char *argv[]) {
	INPUT_TYPE input[INPUT_SIZE];
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
			FUNCTION_NAME(input[i]);
		uint64_t endTime = mach_absolute_time();
		
		double thisTime = intervalInCycles(startTime, endTime);
		bestTime = __builtin_fmin(thisTime, bestTime);
		
		// Move the stack alignment between trials to eliminate (mostly) aliasing effects
		dummyp = alloca(1);
	}
	
	printf("%16s: %f cycles.\n", LIBSTRING, bestTime / (double) INPUT_SIZE);
	
	return 0;
}
