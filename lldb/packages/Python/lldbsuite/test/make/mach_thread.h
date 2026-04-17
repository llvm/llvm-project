#ifdef __APPLE__
#include <mach/mach.h>
#include <pthread.h>
#include <thread>

// Get the Mach thread port for a std::thread. Must be called before join(),
// because the native_handle is invalidated after join.
mach_port_t get_mach_thread(std::thread &t) {
  return pthread_mach_thread_np(t.native_handle());
}

// After join returns, the underlying Mach thread may not have been terminated
// yet (pthread_join uses a semaphore that is signaled before the thread calls
// thread_terminate). If the debugger stops the process in this window it will
// freeze the dying thread, making it appear still alive in task_threads(). Poll
// until the specific Mach thread is gone from the task's thread list.
void wait_for_thread_cleanup(mach_port_t mach_thread) {
  while (true) {
    thread_array_t thread_list;
    mach_msg_type_number_t count;
    kern_return_t kr = task_threads(mach_task_self(), &thread_list, &count);
    if (kr != KERN_SUCCESS)
      break;
    bool found = false;
    for (mach_msg_type_number_t i = 0; i < count; i++) {
      if (thread_list[i] == mach_thread)
        found = true;
      mach_port_deallocate(mach_task_self(), thread_list[i]);
    }
    vm_deallocate(mach_task_self(), (vm_address_t)thread_list,
                  count * sizeof(thread_t));
    if (!found)
      break;
    std::this_thread::yield();
  }
}
#else
#include <thread>
typedef unsigned int mach_port_t;
inline mach_port_t get_mach_thread(std::thread &t) { return 0; }
inline void wait_for_thread_cleanup(mach_port_t) {}
#endif
