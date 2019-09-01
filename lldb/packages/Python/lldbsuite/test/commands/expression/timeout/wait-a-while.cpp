#include <stdio.h>
#include <stdint.h>

#include <chrono>
#include <thread>


int
wait_a_while (int microseconds)
{
    int num_times = 0;
    auto end_time = std::chrono::system_clock::now() + std::chrono::microseconds(microseconds);

    while (1)
    {
        num_times++;
        auto wait_time = end_time - std::chrono::system_clock::now();

        std::this_thread::sleep_for(wait_time);
        if (std::chrono::system_clock::now() > end_time)
            break;
    }
    return num_times;
}

int
main (int argc, char **argv)
{
    printf ("stop here in main.\n");
    int num_times = wait_a_while (argc * 1000);
    printf ("Done, took %d times.\n", num_times);

    return 0;

}
