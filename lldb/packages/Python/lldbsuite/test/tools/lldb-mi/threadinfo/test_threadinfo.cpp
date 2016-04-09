#include <cstdlib>
#include <iostream>
#include <thread>

using namespace std;

void
ThreadProc()
{
    int i = 0;
    i++;
}

int
main()
{
    thread t(ThreadProc);
    t.join();

    return 0;
}
