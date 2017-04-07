//===--------------------- TaskPool.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_TaskPool_h_
#define utility_TaskPool_h_

#include <functional> // for bind, function
#include <future>
#include <list>
#include <memory>      // for make_shared
#include <mutex>       // for mutex, unique_lock, condition_variable
#include <type_traits> // for forward, result_of, move

// Global TaskPool class for running tasks in parallel on a set of worker thread
// created the first
// time the task pool is used. The TaskPool provide no guarantee about the order
// the task will be run
// and about what tasks will run in parallel. None of the task added to the task
// pool should block
// on something (mutex, future, condition variable) what will be set only by the
// completion of an
// other task on the task pool as they may run on the same thread sequentally.
class TaskPool {
public:
  // Add a new task to the task pool and return a std::future belonging to the
  // newly created task.
  // The caller of this function has to wait on the future for this task to
  // complete.
  template <typename F, typename... Args>
  static std::future<typename std::result_of<F(Args...)>::type>
  AddTask(F &&f, Args &&... args);

  // Run all of the specified tasks on the task pool and wait until all of them
  // are finished
  // before returning. This method is intended to be used for small number tasks
  // where listing
  // them as function arguments is acceptable. For running large number of tasks
  // you should use
  // AddTask for each task and then call wait() on each returned future.
  template <typename... T> static void RunTasks(T &&... tasks);

private:
  TaskPool() = delete;

  template <typename... T> struct RunTaskImpl;

  static void AddTaskImpl(std::function<void()> &&task_fn);
};

// Wrapper class around the global TaskPool implementation to make it possible
// to create a set of
// tasks and then wait for the tasks to be completed by the
// WaitForNextCompletedTask call. This
// class should be used when WaitForNextCompletedTask is needed because this
// class add no other
// extra functionality to the TaskPool class and it have a very minor
// performance overhead.
template <typename T> // The return type of the tasks what will be added to this
                      // task runner
                      class TaskRunner {
public:
  // Add a task to the task runner what will also add the task to the global
  // TaskPool. The
  // function doesn't return the std::future for the task because it will be
  // supplied by the
  // WaitForNextCompletedTask after the task is completed.
  template <typename F, typename... Args> void AddTask(F &&f, Args &&... args);

  // Wait for the next task in this task runner to finish and then return the
  // std::future what
  // belongs to the finished task. If there is no task in this task runner
  // (neither pending nor
  // comleted) then this function will return an invalid future. Usually this
  // function should be
  // called in a loop processing the results of the tasks until it returns an
  // invalid std::future
  // what means that all task in this task runner is completed.
  std::future<T> WaitForNextCompletedTask();

  // Convenience method to wait for all task in this TaskRunner to finish. Do
  // NOT use this class
  // just because of this method. Use TaskPool instead and wait for each
  // std::future returned by
  // AddTask in a loop.
  void WaitForAllTasks();

private:
  std::list<std::future<T>> m_ready;
  std::list<std::future<T>> m_pending;
  std::mutex m_mutex;
  std::condition_variable m_cv;
};

template <typename F, typename... Args>
std::future<typename std::result_of<F(Args...)>::type>
TaskPool::AddTask(F &&f, Args &&... args) {
  auto task_sp = std::make_shared<
      std::packaged_task<typename std::result_of<F(Args...)>::type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  AddTaskImpl([task_sp]() { (*task_sp)(); });

  return task_sp->get_future();
}

template <typename... T> void TaskPool::RunTasks(T &&... tasks) {
  RunTaskImpl<T...>::Run(std::forward<T>(tasks)...);
}

template <typename Head, typename... Tail>
struct TaskPool::RunTaskImpl<Head, Tail...> {
  static void Run(Head &&h, Tail &&... t) {
    auto f = AddTask(std::forward<Head>(h));
    RunTaskImpl<Tail...>::Run(std::forward<Tail>(t)...);
    f.wait();
  }
};

template <> struct TaskPool::RunTaskImpl<> {
  static void Run() {}
};

template <typename T>
template <typename F, typename... Args>
void TaskRunner<T>::AddTask(F &&f, Args &&... args) {
  std::unique_lock<std::mutex> lock(m_mutex);
  auto it = m_pending.emplace(m_pending.end());
  *it = std::move(TaskPool::AddTask(
      [this, it](F f, Args... args) {
        T &&r = f(std::forward<Args>(args)...);

        std::unique_lock<std::mutex> lock(this->m_mutex);
        this->m_ready.splice(this->m_ready.end(), this->m_pending, it);
        lock.unlock();

        this->m_cv.notify_one();
        return r;
      },
      std::forward<F>(f), std::forward<Args>(args)...));
}

template <>
template <typename F, typename... Args>
void TaskRunner<void>::AddTask(F &&f, Args &&... args) {
  std::unique_lock<std::mutex> lock(m_mutex);
  auto it = m_pending.emplace(m_pending.end());
  *it = std::move(TaskPool::AddTask(
      [this, it](F f, Args... args) {
        f(std::forward<Args>(args)...);

        std::unique_lock<std::mutex> lock(this->m_mutex);
        this->m_ready.emplace_back(std::move(*it));
        this->m_pending.erase(it);
        lock.unlock();

        this->m_cv.notify_one();
      },
      std::forward<F>(f), std::forward<Args>(args)...));
}

template <typename T> std::future<T> TaskRunner<T>::WaitForNextCompletedTask() {
  std::unique_lock<std::mutex> lock(m_mutex);
  if (m_ready.empty() && m_pending.empty())
    return std::future<T>(); // No more tasks

  if (m_ready.empty())
    m_cv.wait(lock, [this]() { return !this->m_ready.empty(); });

  std::future<T> res = std::move(m_ready.front());
  m_ready.pop_front();

  lock.unlock();
  res.wait();

  return std::move(res);
}

template <typename T> void TaskRunner<T>::WaitForAllTasks() {
  while (WaitForNextCompletedTask().valid())
    ;
}

#endif // #ifndef utility_TaskPool_h_
