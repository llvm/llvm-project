#include <chrono>
#include <thread>
#include <vector>

void *
thread_function (void *thread_marker)
{
    int keep_going = 1; 
    int my_value = *((int *)thread_marker);
    int counter = 0;

    while (counter < 20)
    {
        counter++; // Break here in thread body.
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    return NULL;
}


int 
main ()
{
    std::vector<std::thread> threads;

    int thread_value = 0;
    int i;

    for (i = 0; i < 10; i++)
    {
        thread_value += 1;
        threads.push_back(std::thread(thread_function, &thread_value));
    }

    for (i = 0; i < 10; i++)
        threads[i].join();

    return 0;
}
