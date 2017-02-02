//===-- Timer.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lldb/Core/Timer.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "lldb/Host/Host.h"
#include "lldb/Utility/Stream.h"

#include <stdio.h>

using namespace lldb_private;

#define TIMER_INDENT_AMOUNT 2

namespace {
typedef std::map<const char *, std::chrono::nanoseconds> TimerCategoryMap;
typedef std::vector<Timer *> TimerStack;
} // end of anonymous namespace

std::atomic<bool> Timer::g_quiet(true);
std::atomic<unsigned> Timer::g_display_depth(0);
static std::mutex &GetFileMutex() {
  static std::mutex *g_file_mutex_ptr = new std::mutex();
  return *g_file_mutex_ptr;
}

static std::mutex &GetCategoryMutex() {
  static std::mutex g_category_mutex;
  return g_category_mutex;
}

static TimerCategoryMap &GetCategoryMap() {
  static TimerCategoryMap g_category_map;
  return g_category_map;
}

static void ThreadSpecificCleanup(void *p) {
  delete static_cast<TimerStack *>(p);
}

static TimerStack *GetTimerStackForCurrentThread() {
  static lldb::thread_key_t g_key =
      Host::ThreadLocalStorageCreate(ThreadSpecificCleanup);

  void *timer_stack = Host::ThreadLocalStorageGet(g_key);
  if (timer_stack == NULL) {
    Host::ThreadLocalStorageSet(g_key, new TimerStack);
    timer_stack = Host::ThreadLocalStorageGet(g_key);
  }
  return (TimerStack *)timer_stack;
}

void Timer::SetQuiet(bool value) { g_quiet = value; }

Timer::Timer(const char *category, const char *format, ...)
    : m_category(category), m_total_start(std::chrono::steady_clock::now()) {
  TimerStack *stack = GetTimerStackForCurrentThread();
  if (!stack)
    return;

  stack->push_back(this);
  if (g_quiet && stack->size() <= g_display_depth) {
    std::lock_guard<std::mutex> lock(GetFileMutex());

    // Indent
    ::fprintf(stdout, "%*s", int(stack->size() - 1) * TIMER_INDENT_AMOUNT, "");
    // Print formatted string
    va_list args;
    va_start(args, format);
    ::vfprintf(stdout, format, args);
    va_end(args);

    // Newline
    ::fprintf(stdout, "\n");
  }
}

Timer::~Timer() {
  using namespace std::chrono;

  TimerStack *stack = GetTimerStackForCurrentThread();
  if (!stack)
    return;

  auto stop_time = steady_clock::now();
  auto total_dur = stop_time - m_total_start;
  auto timer_dur = total_dur - m_child_duration;

  if (g_quiet && stack->size() <= g_display_depth) {
    std::lock_guard<std::mutex> lock(GetFileMutex());
    ::fprintf(stdout, "%*s%.9f sec (%.9f sec)\n",
              int(stack->size() - 1) * TIMER_INDENT_AMOUNT, "",
              duration<double>(total_dur).count(),
              duration<double>(timer_dur).count());
  }

  assert(stack->back() == this);
  stack->pop_back();
  if (!stack->empty())
    stack->back()->ChildDuration(total_dur);

  // Keep total results for each category so we can dump results.
  {
    std::lock_guard<std::mutex> guard(GetCategoryMutex());
    TimerCategoryMap &category_map = GetCategoryMap();
    category_map[m_category] += timer_dur;
  }
}

void Timer::SetDisplayDepth(uint32_t depth) { g_display_depth = depth; }

/* binary function predicate:
 * - returns whether a person is less than another person
 */
static bool
CategoryMapIteratorSortCriterion(const TimerCategoryMap::const_iterator &lhs,
                                 const TimerCategoryMap::const_iterator &rhs) {
  return lhs->second > rhs->second;
}

void Timer::ResetCategoryTimes() {
  std::lock_guard<std::mutex> guard(GetCategoryMutex());
  TimerCategoryMap &category_map = GetCategoryMap();
  category_map.clear();
}

void Timer::DumpCategoryTimes(Stream *s) {
  std::lock_guard<std::mutex> guard(GetCategoryMutex());
  TimerCategoryMap &category_map = GetCategoryMap();
  std::vector<TimerCategoryMap::const_iterator> sorted_iterators;
  TimerCategoryMap::const_iterator pos, end = category_map.end();
  for (pos = category_map.begin(); pos != end; ++pos) {
    sorted_iterators.push_back(pos);
  }
  std::sort(sorted_iterators.begin(), sorted_iterators.end(),
            CategoryMapIteratorSortCriterion);

  const size_t count = sorted_iterators.size();
  for (size_t i = 0; i < count; ++i) {
    const auto timer = sorted_iterators[i]->second;
    s->Printf("%.9f sec for %s\n", std::chrono::duration<double>(timer).count(),
              sorted_iterators[i]->first);
  }
}
