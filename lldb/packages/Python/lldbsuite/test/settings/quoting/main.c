#include <stdio.h>
#include <string.h>

/* This program writes its arguments (separated by '\0') to stdout. */
int
main(int argc, char const *argv[])
{
    int i;
    for (i = 1; i < argc; ++i)
        fwrite(argv[i], strlen(argv[i])+1, 1, stdout);

    return 0;
}
