#include <stdlib.h>

int printf(const char * __restrict format, ...);

typedef struct {
    int a;
    int b;
} MYFILE;

int main()
{
    MYFILE *myFile = malloc(sizeof(MYFILE));

    myFile->a = 5;
    myFile->b = 9;

    printf("%d\n", myFile->a + myFile->b); // Set breakpoint 0 here.

    free(myFile);
}
