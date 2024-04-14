#ifndef BS_THREAD_POOL_HPP
#define BS_THREAD_POOL_HPP
/**
 * @file BS_thread_pool.hpp
 * @author Barak Shoshany (baraksh@gmail.com) (https://baraksh.com)
 * @version 4.1.0
 * @date 2024-03-22
 * @copyright Copyright (c) 2024 Barak Shoshany. Licensed under the MIT license. If you found this project useful, please consider starring it on GitHub! If you use this library in software of any kind, please provide a link to the GitHub repository https://github.com/bshoshany/thread-pool in the source code and documentation. If you use this library in published research, please cite it as follows: Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific Computing", doi:10.1016/j.softx.2024.101687, SoftwareX 26 (2024) 101687, arXiv:2105.00613
 *
 * @brief BS::thread_pool: a fast, lightweight, and easy-to-use C++17 thread pool library. This header file contains the main thread pool class and some additional classes and definitions. No other files are needed in order to use the thread pool itself.
 */

#ifndef __cpp_exceptions
    #define BS_THREAD_POOL_DISABLE_EXCEPTION_HANDLING
    #undef BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
#endif

#include <chrono>             // std::chrono
#include <condition_variable> // std::condition_variable
#include <cstddef>            // std::size_t
#ifdef BS_THREAD_POOL_ENABLE_PRIORITY
    #include <cstdint>        // std::int_least16_t
#endif
#ifndef BS_THREAD_POOL_DISABLE_EXCEPTION_HANDLING
    #include <exception>      // std::current_exception
#endif
#include <functional>         // std::function
#include <future>             // std::future, std::future_status, std::promise
#include <memory>             // std::make_shared, std::make_unique, std::shared_ptr, std::unique_ptr
#include <mutex>              // std::mutex, std::scoped_lock, std::unique_lock
#include <optional>           // std::nullopt, std::optional
#include <queue>              // std::priority_queue (if priority enabled), std::queue
#ifdef BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
    #include <stdexcept>      // std::runtime_error
#endif
#include <thread>             // std::thread
#include <type_traits>        // std::conditional_t, std::decay_t, std::invoke_result_t, std::is_void_v, std::remove_const_t (if priority enabled)
#include <utility>            // std::forward, std::move
#include <vector>             // std::vector

/**
 * @brief A namespace used by Barak Shoshany's projects.
 */
namespace BS {
// Macros indicating the version of the thread pool library.
#define BS_THREAD_POOL_VERSION_MAJOR 4
#define BS_THREAD_POOL_VERSION_MINOR 1
#define BS_THREAD_POOL_VERSION_PATCH 0

class thread_pool;

/**
 * @brief A type to represent the size of things.
 */
using size_t = std::size_t;

/**
 * @brief A convenient shorthand for the type of `std::thread::hardware_concurrency()`. Should evaluate to unsigned int.
 */
using concurrency_t = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;

#ifdef BS_THREAD_POOL_ENABLE_PRIORITY
/**
 * @brief A type used to indicate the priority of a task. Defined to be an integer with a width of (at least) 16 bits.
 */
using priority_t = std::int_least16_t;

/**
 * @brief A namespace containing some pre-defined priorities for convenience.
 */
namespace pr {
    constexpr priority_t highest = 32767;
    constexpr priority_t high = 16383;
    constexpr priority_t normal = 0;
    constexpr priority_t low = -16384;
    constexpr priority_t lowest = -32768;
} // namespace pr

    // Macros used internally to enable or disable the priority arguments in the relevant functions.
    #define BS_THREAD_POOL_PRIORITY_INPUT , const priority_t priority = 0
    #define BS_THREAD_POOL_PRIORITY_OUTPUT , priority
#else
    #define BS_THREAD_POOL_PRIORITY_INPUT
    #define BS_THREAD_POOL_PRIORITY_OUTPUT
#endif

/**
 * @brief A namespace used to obtain information about the current thread.
 */
namespace this_thread {
    /**
     * @brief A type returned by `BS::this_thread::get_index()` which can optionally contain the index of a thread, if that thread belongs to a `BS::thread_pool`. Otherwise, it will contain no value.
     */
    using optional_index = std::optional<size_t>;

    /**
     * @brief A type returned by `BS::this_thread::get_pool()` which can optionally contain the pointer to the pool that owns a thread, if that thread belongs to a `BS::thread_pool`. Otherwise, it will contain no value.
     */
    using optional_pool = std::optional<thread_pool*>;

    /**
     * @brief A helper class to store information about the index of the current thread.
     */
    class [[nodiscard]] thread_info_index
    {
        friend class BS::thread_pool;

    public:
        /**
         * @brief Get the index of the current thread. If this thread belongs to a `BS::thread_pool` object, it will have an index from 0 to `BS::thread_pool::get_thread_count() - 1`. Otherwise, for example if this thread is the main thread or an independent `std::thread`, `std::nullopt` will be returned.
         *
         * @return An `std::optional` object, optionally containing a thread index. Unless you are 100% sure this thread is in a pool, first use `std::optional::has_value()` to check if it contains a value, and if so, use `std::optional::value()` to obtain that value.
         */
        [[nodiscard]] optional_index operator()() const
        {
            return index;
        }

    private:
        /**
         * @brief The index of the current thread.
         */
        optional_index index = std::nullopt;
    }; // class thread_info_index

    /**
     * @brief A helper class to store information about the thread pool that owns the current thread.
     */
    class [[nodiscard]] thread_info_pool
    {
        friend class BS::thread_pool;

    public:
        /**
         * @brief Get the pointer to the thread pool that owns the current thread. If this thread belongs to a `BS::thread_pool` object, a pointer to that object will be returned. Otherwise, for example if this thread is the main thread or an independent `std::thread`, `std::nullopt` will be returned.
         *
         * @return An `std::optional` object, optionally containing a pointer to a thread pool. Unless you are 100% sure this thread is in a pool, first use `std::optional::has_value()` to check if it contains a value, and if so, use `std::optional::value()` to obtain that value.
         */
        [[nodiscard]] optional_pool operator()() const
        {
            return pool;
        }

    private:
        /**
         * @brief A pointer to the thread pool that owns the current thread.
         */
        optional_pool pool = std::nullopt;
    }; // class thread_info_pool

    /**
     * @brief A `thread_local` object used to obtain information about the index of the current thread.
     */
    inline thread_local thread_info_index get_index;

    /**
     * @brief A `thread_local` object used to obtain information about the thread pool that owns the current thread.
     */
    inline thread_local thread_info_pool get_pool;
} // namespace this_thread

/**
 * @brief A helper class to facilitate waiting for and/or getting the results of multiple futures at once.
 *
 * @tparam T The return type of the futures.
 */
template <typename T>
class [[nodiscard]] multi_future : public std::vector<std::future<T>>
{
public:
    // Inherit all constructors from the base class `std::vector`.
    using std::vector<std::future<T>>::vector;

    // The copy constructor and copy assignment operator are deleted. The elements stored in a `multi_future` are futures, which cannot be copied.
    multi_future(const multi_future&) = delete;
    multi_future& operator=(const multi_future&) = delete;

    // The move constructor and move assignment operator are defaulted.
    multi_future(multi_future&&) = default;
    multi_future& operator=(multi_future&&) = default;

    /**
     * @brief Get the results from all the futures stored in this `multi_future`, rethrowing any stored exceptions.
     *
     * @return If the futures return `void`, this function returns `void` as well. Otherwise, it returns a vector containing the results.
     */
    [[nodiscard]] std::conditional_t<std::is_void_v<T>, void, std::vector<T>> get()
    {
        if constexpr (std::is_void_v<T>)
        {
            for (std::future<T>& future : *this)
                future.get();
            return;
        }
        else
        {
            std::vector<T> results;
            results.reserve(this->size());
            for (std::future<T>& future : *this)
                results.push_back(future.get());
            return results;
        }
    }

    /**
     * @brief Check how many of the futures stored in this `multi_future` are ready.
     *
     * @return The number of ready futures.
     */
    [[nodiscard]] size_t ready_count() const
    {
        size_t count = 0;
        for (const std::future<T>& future : *this)
        {
            if (future.wait_for(std::chrono::duration<double>::zero()) == std::future_status::ready)
                ++count;
        }
        return count;
    }

    /**
     * @brief Check if all the futures stored in this `multi_future` are valid.
     *
     * @return `true` if all futures are valid, `false` if at least one of the futures is not valid.
     */
    [[nodiscard]] bool valid() const
    {
        bool is_valid = true;
        for (const std::future<T>& future : *this)
            is_valid = is_valid && future.valid();
        return is_valid;
    }

    /**
     * @brief Wait for all the futures stored in this `multi_future`.
     */
    void wait() const
    {
        for (const std::future<T>& future : *this)
            future.wait();
    }

    /**
     * @brief Wait for all the futures stored in this `multi_future`, but stop waiting after the specified duration has passed. This function first waits for the first future for the desired duration. If that future is ready before the duration expires, this function waits for the second future for whatever remains of the duration. It continues similarly until the duration expires.
     *
     * @tparam R An arithmetic type representing the number of ticks to wait.
     * @tparam P An `std::ratio` representing the length of each tick in seconds.
     * @param duration The amount of time to wait.
     * @return `true` if all futures have been waited for before the duration expired, `false` otherwise.
     */
    template <typename R, typename P>
    bool wait_for(const std::chrono::duration<R, P>& duration) const
    {
        const std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
        for (const std::future<T>& future : *this)
        {
            future.wait_for(duration - (std::chrono::steady_clock::now() - start_time));
            if (duration < std::chrono::steady_clock::now() - start_time)
                return false;
        }
        return true;
    }

    /**
     * @brief Wait for all the futures stored in this `multi_future`, but stop waiting after the specified time point has been reached. This function first waits for the first future until the desired time point. If that future is ready before the time point is reached, this function waits for the second future until the desired time point. It continues similarly until the time point is reached.
     *
     * @tparam C The type of the clock used to measure time.
     * @tparam D An `std::chrono::duration` type used to indicate the time point.
     * @param timeout_time The time point at which to stop waiting.
     * @return `true` if all futures have been waited for before the time point was reached, `false` otherwise.
     */
    template <typename C, typename D>
    bool wait_until(const std::chrono::time_point<C, D>& timeout_time) const
    {
        for (const std::future<T>& future : *this)
        {
            future.wait_until(timeout_time);
            if (timeout_time < std::chrono::steady_clock::now())
                return false;
        }
        return true;
    }
}; // class multi_future

/**
 * @brief A fast, lightweight, and easy-to-use C++17 thread pool class.
 */
class [[nodiscard]] thread_pool
{
public:
    // ============================
    // Constructors and destructors
    // ============================

    /**
     * @brief Construct a new thread pool. The number of threads will be the total number of hardware threads available, as reported by the implementation. This is usually determined by the number of cores in the CPU. If a core is hyperthreaded, it will count as two threads.
     */
    thread_pool() : thread_pool(0, [] {}) {}

    /**
     * @brief Construct a new thread pool with the specified number of threads.
     *
     * @param num_threads The number of threads to use.
     */
    explicit thread_pool(const concurrency_t num_threads) : thread_pool(num_threads, [] {}) {}

    /**
     * @brief Construct a new thread pool with the specified initialization function.
     *
     * @param init_task An initialization function to run in each thread before it starts to execute any submitted tasks. The function must take no arguments and have no return value. It will only be executed exactly once, when the thread is first constructed.
     */
    explicit thread_pool(const std::function<void()>& init_task) : thread_pool(0, init_task) {}

    /**
     * @brief Construct a new thread pool with the specified number of threads and initialization function.
     *
     * @param num_threads The number of threads to use.
     * @param init_task An initialization function to run in each thread before it starts to execute any submitted tasks. The function must take no arguments and have no return value. It will only be executed exactly once, when the thread is first constructed.
     */
    thread_pool(const concurrency_t num_threads, const std::function<void()>& init_task) : thread_count(determine_thread_count(num_threads)), threads(std::make_unique<std::thread[]>(determine_thread_count(num_threads)))
    {
        create_threads(init_task);
    }

    // The copy and move constructors and assignment operators are deleted. The thread pool uses a mutex, which cannot be copied or moved.
    thread_pool(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

    /**
     * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads. Note that if the pool is paused, then any tasks still in the queue will never be executed.
     */
    ~thread_pool()
    {
        wait();
        destroy_threads();
    }

    // =======================
    // Public member functions
    // =======================

#ifdef BS_THREAD_POOL_ENABLE_NATIVE_HANDLES
    /**
     * @brief Get a vector containing the underlying implementation-defined thread handles for each of the pool's threads, as obtained by `std::thread::native_handle()`. Only enabled if `BS_THREAD_POOL_ENABLE_NATIVE_HANDLES` is defined.
     *
     * @return The native thread handles.
     */
    [[nodiscard]] std::vector<std::thread::native_handle_type> get_native_handles() const
    {
        std::vector<std::thread::native_handle_type> native_handles(thread_count);
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            native_handles[i] = threads[i].native_handle();
        }
        return native_handles;
    }
#endif

    /**
     * @brief Get the number of tasks currently waiting in the queue to be executed by the threads.
     *
     * @return The number of queued tasks.
     */
    [[nodiscard]] size_t get_tasks_queued() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks.size();
    }

    /**
     * @brief Get the number of tasks currently being executed by the threads.
     *
     * @return The number of running tasks.
     */
    [[nodiscard]] size_t get_tasks_running() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks_running;
    }

    /**
     * @brief Get the total number of unfinished tasks: either still waiting in the queue, or running in a thread. Note that `get_tasks_total() == get_tasks_queued() + get_tasks_running()`.
     *
     * @return The total number of tasks.
     */
    [[nodiscard]] size_t get_tasks_total() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks_running + tasks.size();
    }

    /**
     * @brief Get the number of threads in the pool.
     *
     * @return The number of threads.
     */
    [[nodiscard]] concurrency_t get_thread_count() const
    {
        return thread_count;
    }

    /**
     * @brief Get a vector containing the unique identifiers for each of the pool's threads, as obtained by `std::thread::get_id()`.
     *
     * @return The unique thread identifiers.
     */
    [[nodiscard]] std::vector<std::thread::id> get_thread_ids() const
    {
        std::vector<std::thread::id> thread_ids(thread_count);
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            thread_ids[i] = threads[i].get_id();
        }
        return thread_ids;
    }

#ifdef BS_THREAD_POOL_ENABLE_PAUSE
    /**
     * @brief Check whether the pool is currently paused. Only enabled if `BS_THREAD_POOL_ENABLE_PAUSE` is defined.
     *
     * @return `true` if the pool is paused, `false` if it is not paused.
     */
    [[nodiscard]] bool is_paused() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return paused;
    }

    /**
     * @brief Pause the pool. The workers will temporarily stop retrieving new tasks out of the queue, although any tasks already executed will keep running until they are finished. Only enabled if `BS_THREAD_POOL_ENABLE_PAUSE` is defined.
     */
    void pause()
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        paused = true;
    }
#endif

    /**
     * @brief Purge all the tasks waiting in the queue. Tasks that are currently running will not be affected, but any tasks still waiting in the queue will be discarded, and will never be executed by the threads. Please note that there is no way to restore the purged tasks.
     */
    void purge()
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        while (!tasks.empty())
            tasks.pop();
    }

    /**
     * @brief Submit a function with no arguments and no return value into the task queue, with the specified priority. To push a function with arguments, enclose it in a lambda expression. Does not return a future, so the user must use `wait()` or some other method to ensure that the task finishes executing, otherwise bad things will happen.
     *
     * @tparam F The type of the function.
     * @param task The function to push.
     * @param priority The priority of the task. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     */
    template <typename F>
    void detach_task(F&& task BS_THREAD_POOL_PRIORITY_INPUT)
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            tasks.emplace(std::forward<F>(task) BS_THREAD_POOL_PRIORITY_OUTPUT);
        }
        task_available_cv.notify_one();
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The block function takes two arguments, the start and end of the block, so that it is only called only once per block, but it is up to the user make sure the block function correctly deals with all the indices in each block. Does not return a `multi_future`, so the user must use `wait()` or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     * @tparam F The type of the function to loop through.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no blocks will be submitted.
     * @param block A function that will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. `block(start, end)` should typically involve a loop of the form `for (T i = start; i < end; ++i)`.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     */
    template <typename T, typename F>
    void detach_blocks(const T first_index, const T index_after_last, F&& block, const size_t num_blocks = 0 BS_THREAD_POOL_PRIORITY_INPUT)
    {
        if (index_after_last > first_index)
        {
            const blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            for (size_t blk = 0; blk < blks.get_num_blocks(); ++blk)
                detach_task(
                    [block = std::forward<F>(block), start = blks.start(blk), end = blks.end(blk)]
                    {
                        block(start, end);
                    } BS_THREAD_POOL_PRIORITY_OUTPUT);
        }
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The loop function takes one argument, the loop index, so that it is called many times per block. Does not return a `multi_future`, so the user must use `wait()` or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     * @tparam F The type of the function to loop through.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no blocks will be submitted.
     * @param loop The function to loop through. Will be called once per index, many times per block. Should take exactly one argument: the loop index.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     */
    template <typename T, typename F>
    void detach_loop(const T first_index, const T index_after_last, F&& loop, const size_t num_blocks = 0 BS_THREAD_POOL_PRIORITY_INPUT)
    {
        if (index_after_last > first_index)
        {
            const blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            for (size_t blk = 0; blk < blks.get_num_blocks(); ++blk)
                detach_task(
                    [loop = std::forward<F>(loop), start = blks.start(blk), end = blks.end(blk)]
                    {
                        for (T i = start; i < end; ++i)
                            loop(i);
                    } BS_THREAD_POOL_PRIORITY_OUTPUT);
        }
    }

    /**
     * @brief Submit a sequence of tasks enumerated by indices to the queue, with the specified priority. Does not return a `multi_future`, so the user must use `wait()` or some other method to ensure that the sequence finishes executing, otherwise bad things will happen.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     * @tparam F The type of the function used to define the sequence.
     * @param first_index The first index in the sequence.
     * @param index_after_last The index after the last index in the sequence. The sequence will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted.
     * @param sequence The function used to define the sequence. Will be called once per index. Should take exactly one argument, the index.
     * @param priority The priority of the tasks. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     */
    template <typename T, typename F>
    void detach_sequence(const T first_index, const T index_after_last, F&& sequence BS_THREAD_POOL_PRIORITY_INPUT)
    {
        for (T i = first_index; i < index_after_last; ++i)
            detach_task(
                [sequence = std::forward<F>(sequence), i]
                {
                    sequence(i);
                } BS_THREAD_POOL_PRIORITY_OUTPUT);
    }

    /**
     * @brief Reset the pool with the total number of hardware threads available, as reported by the implementation. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
     */
    void reset()
    {
        reset(0, [] {});
    }

    /**
     * @brief Reset the pool with a new number of threads. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param num_threads The number of threads to use.
     */
    void reset(const concurrency_t num_threads)
    {
        reset(num_threads, [] {});
    }

    /**
     * @brief Reset the pool with the total number of hardware threads available, as reported by the implementation, and a new initialization function. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads and initialization function. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param init_task An initialization function to run in each thread before it starts to execute any submitted tasks. The function must take no arguments and have no return value. It will only be executed exactly once, when the thread is first constructed.
     */
    void reset(const std::function<void()>& init_task)
    {
        reset(0, init_task);
    }

    /**
     * @brief Reset the pool with a new number of threads and a new initialization function. Waits for all currently running tasks to be completed, then destroys all threads in the pool and creates a new thread pool with the new number of threads and initialization function. Any tasks that were waiting in the queue before the pool was reset will then be executed by the new threads. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param num_threads The number of threads to use.
     * @param init_task An initialization function to run in each thread before it starts to execute any submitted tasks. The function must take no arguments and have no return value. It will only be executed exactly once, when the thread is first constructed.
     */
    void reset(const concurrency_t num_threads, const std::function<void()>& init_task)
    {
#ifdef BS_THREAD_POOL_ENABLE_PAUSE
        std::unique_lock tasks_lock(tasks_mutex);
        const bool was_paused = paused;
        paused = true;
        tasks_lock.unlock();
#endif
        wait();
        destroy_threads();
        thread_count = determine_thread_count(num_threads);
        threads = std::make_unique<std::thread[]>(thread_count);
        create_threads(init_task);
#ifdef BS_THREAD_POOL_ENABLE_PAUSE
        tasks_lock.lock();
        paused = was_paused;
#endif
    }

    /**
     * @brief Submit a function with no arguments into the task queue, with the specified priority. To submit a function with arguments, enclose it in a lambda expression. If the function has a return value, get a future for the eventual returned value. If the function has no return value, get an `std::future<void>` which can be used to wait until the task finishes.
     *
     * @tparam F The type of the function.
     * @tparam R The return type of the function (can be `void`).
     * @param task The function to submit.
     * @param priority The priority of the task. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     * @return A future to be used later to wait for the function to finish executing and/or obtain its returned value if it has one.
     */
    template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
    [[nodiscard]] std::future<R> submit_task(F&& task BS_THREAD_POOL_PRIORITY_INPUT)
    {
        const std::shared_ptr<std::promise<R>> task_promise = std::make_shared<std::promise<R>>();
        detach_task(
            [task = std::forward<F>(task), task_promise]
            {
#ifndef BS_THREAD_POOL_DISABLE_EXCEPTION_HANDLING
                try
                {
#endif
                    if constexpr (std::is_void_v<R>)
                    {
                        task();
                        task_promise->set_value();
                    }
                    else
                    {
                        task_promise->set_value(task());
                    }
#ifndef BS_THREAD_POOL_DISABLE_EXCEPTION_HANDLING
                }
                catch (...)
                {
                    try
                    {
                        task_promise->set_exception(std::current_exception());
                    }
                    catch (...)
                    {
                    }
                }
#endif
            } BS_THREAD_POOL_PRIORITY_OUTPUT);
        return task_promise->get_future();
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The block function takes two arguments, the start and end of the block, so that it is only called only once per block, but it is up to the user make sure the block function correctly deals with all the indices in each block. Returns a `multi_future` that contains the futures for all of the blocks.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     * @tparam F The type of the function to loop through.
     * @tparam R The return type of the function to loop through (can be `void`).
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no blocks will be submitted, and an empty `multi_future` will be returned.
     * @param block A function that will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. `block(start, end)` should typically involve a loop of the form `for (T i = start; i < end; ++i)`.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     * @return A `multi_future` that can be used to wait for all the blocks to finish. If the block function returns a value, the `multi_future` can also be used to obtain the values returned by each block.
     */
    template <typename T, typename F, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
    [[nodiscard]] multi_future<R> submit_blocks(const T first_index, const T index_after_last, F&& block, const size_t num_blocks = 0 BS_THREAD_POOL_PRIORITY_INPUT)
    {
        if (index_after_last > first_index)
        {
            const blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            multi_future<R> future;
            future.reserve(blks.get_num_blocks());
            for (size_t blk = 0; blk < blks.get_num_blocks(); ++blk)
                future.push_back(submit_task(
                    [block = std::forward<F>(block), start = blks.start(blk), end = blks.end(blk)]
                    {
                        return block(start, end);
                    } BS_THREAD_POOL_PRIORITY_OUTPUT));
            return future;
        }
        return {};
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The loop function takes one argument, the loop index, so that it is called many times per block. It must have no return value. Returns a `multi_future` that contains the futures for all of the blocks.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     * @tparam F The type of the function to loop through.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted, and an empty `multi_future` will be returned.
     * @param loop The function to loop through. Will be called once per index, many times per block. Should take exactly one argument: the loop index. It cannot have a return value.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     * @return A `multi_future` that can be used to wait for all the blocks to finish.
     */
    template <typename T, typename F>
    [[nodiscard]] multi_future<void> submit_loop(const T first_index, const T index_after_last, F&& loop, const size_t num_blocks = 0 BS_THREAD_POOL_PRIORITY_INPUT)
    {
        if (index_after_last > first_index)
        {
            const blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            multi_future<void> future;
            future.reserve(blks.get_num_blocks());
            for (size_t blk = 0; blk < blks.get_num_blocks(); ++blk)
                future.push_back(submit_task(
                    [loop = std::forward<F>(loop), start = blks.start(blk), end = blks.end(blk)]
                    {
                        for (T i = start; i < end; ++i)
                            loop(i);
                    } BS_THREAD_POOL_PRIORITY_OUTPUT));
            return future;
        }
        return {};
    }

    /**
     * @brief Submit a sequence of tasks enumerated by indices to the queue, with the specified priority. Returns a `multi_future` that contains the futures for all of the tasks.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     * @tparam F The type of the function used to define the sequence.
     * @tparam R The return type of the function used to define the sequence (can be `void`).
     * @param first_index The first index in the sequence.
     * @param index_after_last The index after the last index in the sequence. The sequence will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted, and an empty `multi_future` will be returned.
     * @param sequence The function used to define the sequence. Will be called once per index. Should take exactly one argument, the index.
     * @param priority The priority of the tasks. Should be between -32,768 and 32,767 (a signed 16-bit integer). The default is 0. Only enabled if `BS_THREAD_POOL_ENABLE_PRIORITY` is defined.
     * @return A `multi_future` that can be used to wait for all the tasks to finish. If the sequence function returns a value, the `multi_future` can also be used to obtain the values returned by each task.
     */
    template <typename T, typename F, typename R = std::invoke_result_t<std::decay_t<F>, T>>
    [[nodiscard]] multi_future<R> submit_sequence(const T first_index, const T index_after_last, F&& sequence BS_THREAD_POOL_PRIORITY_INPUT)
    {
        if (index_after_last > first_index)
        {
            multi_future<R> future;
            future.reserve(static_cast<size_t>(index_after_last - first_index));
            for (T i = first_index; i < index_after_last; ++i)
                future.push_back(submit_task(
                    [sequence = std::forward<F>(sequence), i]
                    {
                        return sequence(i);
                    } BS_THREAD_POOL_PRIORITY_OUTPUT));
            return future;
        }
        return {};
    }

#ifdef BS_THREAD_POOL_ENABLE_PAUSE
    /**
     * @brief Unpause the pool. The workers will resume retrieving new tasks out of the queue. Only enabled if `BS_THREAD_POOL_ENABLE_PAUSE` is defined.
     */
    void unpause()
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            paused = false;
        }
        task_available_cv.notify_all();
    }
#endif

// Macros used internally to enable or disable pausing in the waiting and worker functions.
#ifdef BS_THREAD_POOL_ENABLE_PAUSE
    #define BS_THREAD_POOL_PAUSED_OR_EMPTY (paused || tasks.empty())
#else
    #define BS_THREAD_POOL_PAUSED_OR_EMPTY tasks.empty()
#endif

    /**
     * @brief Wait for tasks to be completed. Normally, this function waits for all tasks, both those that are currently running in the threads and those that are still waiting in the queue. However, if the pool is paused, this function only waits for the currently running tasks (otherwise it would wait forever). Note: To wait for just one specific task, use `submit_task()` instead, and call the `wait()` member function of the generated future.
     *
     * @throws `wait_deadlock` if called from within a thread of the same pool, which would result in a deadlock. Only enabled if `BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK` is defined.
     */
    void wait()
    {
#ifdef BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
        if (this_thread::get_pool() == this)
            throw wait_deadlock();
#endif
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        tasks_done_cv.wait(tasks_lock,
            [this]
            {
                return (tasks_running == 0) && BS_THREAD_POOL_PAUSED_OR_EMPTY;
            });
        waiting = false;
    }

    /**
     * @brief Wait for tasks to be completed, but stop waiting after the specified duration has passed.
     *
     * @tparam R An arithmetic type representing the number of ticks to wait.
     * @tparam P An `std::ratio` representing the length of each tick in seconds.
     * @param duration The amount of time to wait.
     * @return `true` if all tasks finished running, `false` if the duration expired but some tasks are still running.
     *
     * @throws `wait_deadlock` if called from within a thread of the same pool, which would result in a deadlock. Only enabled if `BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK` is defined.
     */
    template <typename R, typename P>
    bool wait_for(const std::chrono::duration<R, P>& duration)
    {
#ifdef BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
        if (this_thread::get_pool() == this)
            throw wait_deadlock();
#endif
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        const bool status = tasks_done_cv.wait_for(tasks_lock, duration,
            [this]
            {
                return (tasks_running == 0) && BS_THREAD_POOL_PAUSED_OR_EMPTY;
            });
        waiting = false;
        return status;
    }

    /**
     * @brief Wait for tasks to be completed, but stop waiting after the specified time point has been reached.
     *
     * @tparam C The type of the clock used to measure time.
     * @tparam D An `std::chrono::duration` type used to indicate the time point.
     * @param timeout_time The time point at which to stop waiting.
     * @return `true` if all tasks finished running, `false` if the time point was reached but some tasks are still running.
     *
     * @throws `wait_deadlock` if called from within a thread of the same pool, which would result in a deadlock. Only enabled if `BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK` is defined.
     */
    template <typename C, typename D>
    bool wait_until(const std::chrono::time_point<C, D>& timeout_time)
    {
#ifdef BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
        if (this_thread::get_pool() == this)
            throw wait_deadlock();
#endif
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        const bool status = tasks_done_cv.wait_until(tasks_lock, timeout_time,
            [this]
            {
                return (tasks_running == 0) && BS_THREAD_POOL_PAUSED_OR_EMPTY;
            });
        waiting = false;
        return status;
    }

#ifdef BS_THREAD_POOL_ENABLE_WAIT_DEADLOCK_CHECK
    // ==============
    // Public classes
    // ==============

    /**
     * @brief An exception that will be thrown by `wait()`, `wait_for()`, and `wait_until()` if the user tries to call them from within a thread of the same pool, which would result in a deadlock.
     */
    struct wait_deadlock : public std::runtime_error
    {
        wait_deadlock() : std::runtime_error("BS::thread_pool::wait_deadlock"){};
    };
#endif

private:
    // ========================
    // Private member functions
    // ========================

    /**
     * @brief Create the threads in the pool and assign a worker to each thread.
     *
     * @param init_task An initialization function to run in each thread before it starts to execute any submitted tasks.
     */
    void create_threads(const std::function<void()>& init_task)
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            tasks_running = thread_count;
            workers_running = true;
        }
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            threads[i] = std::thread(&thread_pool::worker, this, i, init_task);
        }
    }

    /**
     * @brief Destroy the threads in the pool.
     */
    void destroy_threads()
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            workers_running = false;
        }
        task_available_cv.notify_all();
        for (concurrency_t i = 0; i < thread_count; ++i)
        {
            threads[i].join();
        }
    }

    /**
     * @brief Determine how many threads the pool should have, based on the parameter passed to the constructor or reset().
     *
     * @param num_threads The parameter passed to the constructor or `reset()`. If the parameter is a positive number, then the pool will be created with this number of threads. If the parameter is non-positive, or a parameter was not supplied (in which case it will have the default value of 0), then the pool will be created with the total number of hardware threads available, as obtained from `std::thread::hardware_concurrency()`. If the latter returns zero for some reason, then the pool will be created with just one thread.
     * @return The number of threads to use for constructing the pool.
     */
    [[nodiscard]] static concurrency_t determine_thread_count(const concurrency_t num_threads)
    {
        if (num_threads > 0)
            return num_threads;
        if (std::thread::hardware_concurrency() > 0)
            return std::thread::hardware_concurrency();
        return 1;
    }

    /**
     * @brief A worker function to be assigned to each thread in the pool. Waits until it is notified by `detach_task()` that a task is available, and then retrieves the task from the queue and executes it. Once the task finishes, the worker notifies `wait()` in case it is waiting.
     *
     * @param idx The index of this thread.
     * @param init_task An initialization function to run in this thread before it starts to execute any submitted tasks.
     */
    void worker(const concurrency_t idx, const std::function<void()>& init_task)
    {
        this_thread::get_index.index = idx;
        this_thread::get_pool.pool = this;
        init_task();
        std::unique_lock tasks_lock(tasks_mutex);
        while (true)
        {
            --tasks_running;
            tasks_lock.unlock();
            if (waiting && (tasks_running == 0) && BS_THREAD_POOL_PAUSED_OR_EMPTY)
                tasks_done_cv.notify_all();
            tasks_lock.lock();
            task_available_cv.wait(tasks_lock,
                [this]
                {
                    return !BS_THREAD_POOL_PAUSED_OR_EMPTY || !workers_running;
                });
            if (!workers_running)
                break;
            {
#ifdef BS_THREAD_POOL_ENABLE_PRIORITY
                const std::function<void()> task = std::move(std::remove_const_t<pr_task&>(tasks.top()).task);
                tasks.pop();
#else
                const std::function<void()> task = std::move(tasks.front());
                tasks.pop();
#endif
                ++tasks_running;
                tasks_lock.unlock();
                task();
            }
            tasks_lock.lock();
        }
        this_thread::get_index.index = std::nullopt;
        this_thread::get_pool.pool = std::nullopt;
    }

    // ===============
    // Private classes
    // ===============

    /**
     * @brief A helper class to divide a range into blocks. Used by `detach_blocks()`, `submit_blocks()`, `detach_loop()`, and `submit_loop()`.
     *
     * @tparam T The type of the indices. Should be a signed or unsigned integer.
     */
    template <typename T>
    class [[nodiscard]] blocks
    {
    public:
        /**
         * @brief Construct a `blocks` object with the given specifications.
         *
         * @param first_index_ The first index in the range.
         * @param index_after_last_ The index after the last index in the range.
         * @param num_blocks_ The desired number of blocks to divide the range into.
         */
        blocks(const T first_index_, const T index_after_last_, const size_t num_blocks_) : first_index(first_index_), index_after_last(index_after_last_), num_blocks(num_blocks_)
        {
            if (index_after_last > first_index)
            {
                const size_t total_size = static_cast<size_t>(index_after_last - first_index);
                if (num_blocks > total_size)
                    num_blocks = total_size;
                block_size = total_size / num_blocks;
                remainder = total_size % num_blocks;
                if (block_size == 0)
                {
                    block_size = 1;
                    num_blocks = (total_size > 1) ? total_size : 1;
                }
            }
            else
            {
                num_blocks = 0;
            }
        }

        /**
         * @brief Get the first index of a block.
         *
         * @param block The block number.
         * @return The first index.
         */
        [[nodiscard]] T start(const size_t block) const
        {
            return first_index + static_cast<T>(block * block_size) + static_cast<T>(block < remainder ? block : remainder);
        }

        /**
         * @brief Get the index after the last index of a block.
         *
         * @param block The block number.
         * @return The index after the last index.
         */
        [[nodiscard]] T end(const size_t block) const
        {
            return (block == num_blocks - 1) ? index_after_last : start(block + 1);
        }

        /**
         * @brief Get the number of blocks. Note that this may be different than the desired number of blocks that was passed to the constructor.
         *
         * @return The number of blocks.
         */
        [[nodiscard]] size_t get_num_blocks() const
        {
            return num_blocks;
        }

    private:
        /**
         * @brief The size of each block (except possibly the last block).
         */
        size_t block_size = 0;

        /**
         * @brief The first index in the range.
         */
        T first_index = 0;

        /**
         * @brief The index after the last index in the range.
         */
        T index_after_last = 0;

        /**
         * @brief The number of blocks.
         */
        size_t num_blocks = 0;

        /**
         * @brief The remainder obtained after dividing the total size by the number of blocks.
         */
        size_t remainder = 0;
    }; // class blocks

#ifdef BS_THREAD_POOL_ENABLE_PRIORITY
    /**
     * @brief A helper class to store a task with an assigned priority.
     */
    class [[nodiscard]] pr_task
    {
        friend class thread_pool;

    public:
        /**
         * @brief Construct a new task with an assigned priority by copying the task.
         *
         * @param task_ The task.
         * @param priority_ The desired priority.
         */
        explicit pr_task(const std::function<void()>& task_, const priority_t priority_ = 0) : task(task_), priority(priority_) {}

        /**
         * @brief Construct a new task with an assigned priority by moving the task.
         *
         * @param task_ The task.
         * @param priority_ The desired priority.
         */
        explicit pr_task(std::function<void()>&& task_, const priority_t priority_ = 0) : task(std::move(task_)), priority(priority_) {}

        /**
         * @brief Compare the priority of two tasks.
         *
         * @param lhs The first task.
         * @param rhs The second task.
         * @return `true` if the first task has a lower priority than the second task, `false` otherwise.
         */
        [[nodiscard]] friend bool operator<(const pr_task& lhs, const pr_task& rhs)
        {
            return lhs.priority < rhs.priority;
        }

    private:
        /**
         * @brief The task.
         */
        std::function<void()> task = {};

        /**
         * @brief The priority of the task.
         */
        priority_t priority = 0;
    }; // class pr_task
#endif

    // ============
    // Private data
    // ============

#ifdef BS_THREAD_POOL_ENABLE_PAUSE
    /**
     * @brief A flag indicating whether the workers should pause. When set to `true`, the workers temporarily stop retrieving new tasks out of the queue, although any tasks already executed will keep running until they are finished. When set to `false` again, the workers resume retrieving tasks.
     */
    bool paused = false;
#endif

    /**
     * @brief A condition variable to notify `worker()` that a new task has become available.
     */
    std::condition_variable task_available_cv = {};

    /**
     * @brief A condition variable to notify `wait()` that the tasks are done.
     */
    std::condition_variable tasks_done_cv = {};

    /**
     * @brief A queue of tasks to be executed by the threads.
     */
#ifdef BS_THREAD_POOL_ENABLE_PRIORITY
    std::priority_queue<pr_task> tasks = {};
#else
    std::queue<std::function<void()>> tasks = {};
#endif

    /**
     * @brief A counter for the total number of currently running tasks.
     */
    size_t tasks_running = 0;

    /**
     * @brief A mutex to synchronize access to the task queue by different threads.
     */
    mutable std::mutex tasks_mutex = {};

    /**
     * @brief The number of threads in the pool.
     */
    concurrency_t thread_count = 0;

    /**
     * @brief A smart pointer to manage the memory allocated for the threads.
     */
    std::unique_ptr<std::thread[]> threads = nullptr;

    /**
     * @brief A flag indicating that `wait()` is active and expects to be notified whenever a task is done.
     */
    bool waiting = false;

    /**
     * @brief A flag indicating to the workers to keep running. When set to `false`, the workers terminate permanently.
     */
    bool workers_running = false;
}; // class thread_pool
} // namespace BS
#endif
