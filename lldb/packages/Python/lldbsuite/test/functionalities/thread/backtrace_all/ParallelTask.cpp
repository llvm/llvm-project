#include <cstdint>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <iostream>
#include <cassert>

class TaskPoolImpl
{
public:
    TaskPoolImpl(uint32_t num_threads) :
        m_stop(false)
    {
        for (uint32_t i = 0; i < num_threads; ++i)
            m_threads.emplace_back(Worker, this);
    }

    ~TaskPoolImpl()
    {
        Stop();
    }

    template<typename F, typename... Args>
    std::future<typename std::result_of<F(Args...)>::type>
    AddTask(F&& f, Args&&... args)
    {
        auto task = std::make_shared<std::packaged_task<typename std::result_of<F(Args...)>::type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::unique_lock<std::mutex> lock(m_tasks_mutex);
        assert(!m_stop && "Can't add task to TaskPool after it is stopped");
        m_tasks.emplace([task](){ (*task)(); });
        lock.unlock();
        m_tasks_cv.notify_one();

        return task->get_future();
    }

    void
    Stop()
    {
        std::unique_lock<std::mutex> lock(m_tasks_mutex);
        m_stop = true;
        m_tasks_mutex.unlock();
        m_tasks_cv.notify_all();
        for (auto& t : m_threads)
            t.join();
    }

private:
    static void
    Worker(TaskPoolImpl* pool)
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(pool->m_tasks_mutex);
            if (pool->m_tasks.empty())
                pool->m_tasks_cv.wait(lock, [pool](){ return !pool->m_tasks.empty() || pool->m_stop; });
            if (pool->m_tasks.empty())
                break;

            std::function<void()> f = pool->m_tasks.front();
            pool->m_tasks.pop();
            lock.unlock();

            f();
        }
    }

    std::queue<std::function<void()>> m_tasks;
    std::mutex                        m_tasks_mutex;
    std::condition_variable           m_tasks_cv;
    bool                              m_stop;
    std::vector<std::thread>          m_threads;
};

class TaskPool
{
public:
    // Add a new task to the thread pool and return a std::future belongs for the newly created task.
    // The caller of this function have to wait on the future for this task to complete.
    template<typename F, typename... Args>
    static std::future<typename std::result_of<F(Args...)>::type>
    AddTask(F&& f, Args&&... args)
    {
        return GetImplementation().AddTask(std::forward<F>(f), std::forward<Args>(args)...);
    }

    // Run all of the specified tasks on the thread pool and wait until all of them are finished
    // before returning
    template<typename... T>
    static void
    RunTasks(T&&... t)
    {
        RunTaskImpl<T...>::Run(std::forward<T>(t)...);
    }

private:
    static TaskPoolImpl&
    GetImplementation()
    {
        static TaskPoolImpl g_task_pool_impl(std::thread::hardware_concurrency());
        return g_task_pool_impl;
    }

    template<typename... T>
    struct RunTaskImpl;
};

template<typename H, typename... T>
struct TaskPool::RunTaskImpl<H, T...>
{
    static void
    Run(H&& h, T&&... t)
    {
        auto f = AddTask(std::forward<H>(h));
        RunTaskImpl<T...>::Run(std::forward<T>(t)...);
        f.wait();
    }
};

template<>
struct TaskPool::RunTaskImpl<>
{
    static void
    Run() {}
};

int main()
{
    std::vector<std::future<uint32_t>> tasks;
    for (int i = 0; i < 100000; ++i)
    {
        tasks.emplace_back(TaskPool::AddTask([](int i){
            uint32_t s = 0;
            for (int j = 0; j <= i; ++j)
                s += j;
            return s;
        },
        i));
    }

    for (auto& it : tasks)  // Set breakpoint here
        it.wait();

    TaskPool::RunTasks(
        []() { return 1; },
        []() { return "aaaa"; }
    );
}
