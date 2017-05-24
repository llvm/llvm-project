#include <condition_variable>
#include <mutex>
#include <thread>

std::mutex mutex;
std::condition_variable cond;

void *
thread3(void *input)
{
    std::unique_lock<std::mutex> lock(mutex);
    cond.notify_all(); // Set break point at this line.
    return NULL;
}

void *
thread2(void *input)
{
    std::unique_lock<std::mutex> lock(mutex);
    cond.notify_all();
    cond.wait(lock);
    return NULL;
}

void *
thread1(void *input)
{
    std::thread thread_2(thread2, nullptr);
    thread_2.join();

    return NULL;
}

int main()
{
    std::unique_lock<std::mutex> lock(mutex);

    std::thread thread_1(thread1, nullptr);
    cond.wait(lock);

    std::thread thread_3(thread3, nullptr);
    cond.wait(lock);

    lock.unlock();

    thread_1.join();
    thread_3.join();

    return 0;
}
