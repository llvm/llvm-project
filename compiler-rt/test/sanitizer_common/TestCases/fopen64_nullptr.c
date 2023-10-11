// Check that fopen64(NULL, "r") is ok.
// `-m32` and `-D_FILE_OFFSET_BITS=64` will make fopen() call fopen64()

// REQUIRES: linux
#include <stdio.h>
FILE * fopen64 ( const char * filename, const char * mode );
const char *fn = NULL;
FILE *f;
int main() { f = fopen64(fn, "r"); }
