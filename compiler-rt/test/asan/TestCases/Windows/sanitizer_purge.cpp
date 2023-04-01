// UNSUPPORTED: target={{.*-windows-gnu}}

#include <Windows.h>
#include <stdio.h>
#include <sanitizer/allocator_interface.h>
#include <psapi.h>

// RUN: %clang_cl_asan -Od %s -Fe%t 
// RUN: %t
// REQUIRES: asan-64-bits

size_t GetRSS() {
  PROCESS_MEMORY_COUNTERS counters;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return 0;
  return counters.WorkingSetSize;
}

int main(){
    for (int i = 0; i < 1000; i++) {
        void* a = malloc(1000);
        free(a);
    }
    size_t rss_pre  = GetRSS();
    __sanitizer_purge_allocator();
    size_t rss_post = GetRSS();

    if (rss_pre <= rss_post){
        return -1;
    }

    return 0;
}