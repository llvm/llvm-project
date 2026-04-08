#ifdef __APPLE__
#include <mach/mach.h>
#include <thread>
#endif

// After join returns, the underlying Mach thread may not have been terminated
// yet (pthread_join uses a semaphore that is signaled before the thread calls
// thread_terminate). If the debugger stops the process in this window it will
// freeze the dying thread, making it appear still alive in task_threads(). Poll
// until the kernel-level thread count matches.
void wait_for_thread_cleanup(unsigned int expected) {
#ifdef __APPLE__
  while (true) {
    thread_array_t thread_list;
    mach_msg_type_number_t count;
    kern_return_t kr = task_threads(mach_task_self(), &thread_list, &count);
    if (kr == KERN_SUCCESS) {
      vm_deallocate(mach_task_self(), (vm_address_t)thread_list,
                    count * sizeof(thread_t));
      if (count <= expected)
        break;
    }
    std::this_thread::yield();
  }
#endif
}
