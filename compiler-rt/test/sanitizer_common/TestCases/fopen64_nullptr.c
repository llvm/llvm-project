// Check that fopen64(NULL, "r") is ok.

// REQUIRES: linux
#include <stdio.h>
FILE * fopen64 ( const char * filename, const char * mode );
const char *fn = NULL;
FILE *f;
int main() { f = fopen64(fn, "r"); }
