#include <chrono>
#include <thread>

void
thread_function ()
{
    // Set thread-specific breakpoint here.
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}

int 
main ()
{
    // Set main breakpoint here.
    std::thread t(thread_function);
    t.join();

    thread_function();
    return 0;
}
