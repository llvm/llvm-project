#include <stdio.h>
#include <fcntl.h>

#include <chrono>
#include <thread>

extern void dont_strip_me()
{
  printf("I wasn't stripped\n");
}

static void *a_function()
{
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100)); 
        dont_strip_me();
    }
    return 0;
}

int main(int argc, char const *argv[])
{
    a_function();
    return 0;
}
