#include <stdio.h>
#include <unistd.h>

int main(int argc, char const *argv[])
{
    lldb_enable_attach();

    printf("pid = %i\n", getpid());
    sleep(10);
    return 0; // breakpoint 1
}
