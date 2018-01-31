#include <MyFramework/MyFramework.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int setup_is_complete = 0;

int main(int argc, const char **argv)
{
    char command[8192];
    sprintf (command,
             "/bin/rm -rf %s/MyFramework %s/MyFramework.framework %s/MyFramework.framework.dSYM",
             argv[1], argv[1], argv[1]);
    system (command);

    setup_is_complete = 1;

    // At this point we want lldb to attach to the process.  If lldb attaches
    // before we've removed the framework we're running against, it will be
    // easy for lldb to find the binary & dSYM without using target.exec-search-paths,
    // which is the point of this test.

    for (int loop_limiter = 0; loop_limiter < 100; loop_limiter++)
        sleep (1);

     return foo();
}
