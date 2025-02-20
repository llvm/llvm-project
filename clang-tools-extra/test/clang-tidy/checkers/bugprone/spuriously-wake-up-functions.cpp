// RUN: %check_clang_tidy %s bugprone-spuriously-wake-up-functions %t -- --
#define NULL 0

namespace std {
using intmax_t = int;

template <intmax_t N, intmax_t D = 1>
class ratio {
public:
  static constexpr intmax_t num = 0;
  static constexpr intmax_t den = 0;
  typedef ratio<num, den> type;
};
typedef ratio<1, 1000> milli;
namespace chrono {

template <class Rep, class Period = ratio<1>>
class duration {
public:
  using rep = Rep;
  using period = Period;

public:
  constexpr duration() = default;
  template <class Rep2>
  constexpr explicit duration(const Rep2 &r);
  template <class Rep2, class Period2>
  constexpr duration(const duration<Rep2, Period2> &d);
  ~duration() = default;
  duration(const duration &) = default;
};

template <class Clock, class Duration = typename Clock::duration>
class time_point {
public:
  using clock = Clock;
  using duration = Duration;

public:
  constexpr time_point();
  constexpr explicit time_point(const duration &d);
  template <class Duration2>
  constexpr time_point(const time_point<clock, Duration2> &t);
};

using milliseconds = duration<int, milli>;

class system_clock {
public:
  typedef milliseconds duration;
  typedef duration::rep rep;
  typedef duration::period period;
  typedef chrono::time_point<system_clock> time_point;

  static time_point now() noexcept;
};
} // namespace chrono

class mutex;
template <class Mutex>
class unique_lock {
public:
  typedef Mutex mutex_type;

  unique_lock() noexcept;
  explicit unique_lock(mutex_type &m);
};

class mutex {
public:
  constexpr mutex() noexcept;
  ~mutex();
  mutex(const mutex &) = delete;
  mutex &operator=(const mutex &) = delete;
};

enum class cv_status {
  no_timeout,
  timeout
};

class condition_variable {
public:
  condition_variable();
  ~condition_variable();
  condition_variable(const condition_variable &) = delete;

  void wait(unique_lock<mutex> &lock);
  template <class Predicate>
  void wait(unique_lock<mutex> &lock, Predicate pred);
  template <class Clock, class Duration>
  cv_status wait_until(unique_lock<mutex> &lock,
                       const chrono::time_point<Clock, Duration> &abs_time){};
  template <class Clock, class Duration, class Predicate>
  bool wait_until(unique_lock<mutex> &lock,
                  const chrono::time_point<Clock, Duration> &abs_time,
                  Predicate pred){};
  template <class Rep, class Period>
  cv_status wait_for(unique_lock<mutex> &lock,
                     const chrono::duration<Rep, Period> &rel_time){};
  template <class Rep, class Period, class Predicate>
  bool wait_for(unique_lock<mutex> &lock,
                const chrono::duration<Rep, Period> &rel_time,
                Predicate pred){};
};

} // namespace std

struct Node1 {
  void *Node1;
  struct Node1 *next;
};

static Node1 list;
static std::mutex m;
static std::condition_variable condition;

void consume_list_element(std::condition_variable &condition) {
  std::unique_lock<std::mutex> lk(m);

  if (list.next == nullptr) {
    condition.wait(lk);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'wait' should be placed inside a while statement or used with a conditional parameter [bugprone-spuriously-wake-up-functions]
  }

  while (list.next == nullptr) {
    condition.wait(lk);
  }

  do {
    condition.wait(lk);
  } while (list.next == nullptr);

  for (;; list.next == nullptr) {
    condition.wait(lk);
  }

  if (list.next == nullptr) {
    while (list.next == nullptr) {
      condition.wait(lk);
    }
  }

  if (list.next == nullptr) {
    do {
      condition.wait(lk);
    } while (list.next == nullptr);
  }

  if (list.next == nullptr) {
    for (;; list.next == nullptr) {
      condition.wait(lk);
    }
  }
  using durtype = std::chrono::duration<int, std::milli>;
  durtype dur = std::chrono::duration<int, std::milli>();
  if (list.next == nullptr) {
    condition.wait_for(lk, dur);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'wait_for' should be placed inside a while statement or used with a conditional parameter [bugprone-spuriously-wake-up-functions]
  }
  if (list.next == nullptr) {
    condition.wait_for(lk, dur, [] { return 1; });
  }
  while (list.next == nullptr) {
    condition.wait_for(lk, dur);
  }
  do {
    condition.wait_for(lk, dur);
  } while (list.next == nullptr);
  for (;; list.next == nullptr) {
    condition.wait_for(lk, dur);
  }

  auto now = std::chrono::system_clock::now();
  if (list.next == nullptr) {
    condition.wait_until(lk, now);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'wait_until' should be placed inside a while statement or used with a conditional parameter [bugprone-spuriously-wake-up-functions]
  }
  if (list.next == nullptr) {
    condition.wait_until(lk, now, [] { return 1; });
  }
  while (list.next == nullptr) {
    condition.wait_until(lk, now);
  }
  do {
    condition.wait_until(lk, now);
  } while (list.next == nullptr);
  for (;; list.next == nullptr) {
    condition.wait_until(lk, now);
  }
}
