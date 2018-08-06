#include <stdio.h>
#ifdef _MSC_VER
#include <windows.h>
#define sleep(x) Sleep((x) * 1000)
#else
#include <unistd.h>
#endif

int main(int argc, char const *argv[])
{
    lldb_enable_attach();

    printf("Hello world.\n"); // Set break point at this line.
    if (argc == 1)
        return 0;

    // Waiting to be attached by the debugger, otherwise.
    char line[100];
    while (1) 
        sleep (1); // Waiting to be attached...

    printf("Exiting now\n");
}
