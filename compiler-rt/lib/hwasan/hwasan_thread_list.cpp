#include "hwasan_thread_list.h"

#include "sanitizer_common/sanitizer_thread_arg_retval.h"

namespace __hwasan {

static HwasanThreadList *hwasan_thread_list;
static ThreadArgRetval *thread_data;

HwasanThreadList &hwasanThreadList() { return *hwasan_thread_list; }
ThreadArgRetval &hwasanThreadArgRetval() { return *thread_data; }

void InitThreadList(uptr storage, uptr size) {
  CHECK_EQ(hwasan_thread_list, nullptr);

  static ALIGNED(alignof(
      HwasanThreadList)) char thread_list_placeholder[sizeof(HwasanThreadList)];
  hwasan_thread_list =
      new (thread_list_placeholder) HwasanThreadList(storage, size);

  CHECK_EQ(thread_data, nullptr);

  static ALIGNED(alignof(
      ThreadArgRetval)) char thread_data_placeholder[sizeof(ThreadArgRetval)];
  thread_data = new (thread_data_placeholder) ThreadArgRetval();
}

}  // namespace __hwasan
