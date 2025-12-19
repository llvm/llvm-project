// Phase 4 Benchmark: Performance comparison between old and fast algorithms
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <chrono>

// Extracted constants from time_constants.h
namespace time_constants {
  constexpr int SECONDS_PER_MIN = 60;
  constexpr int SECONDS_PER_HOUR = 3600;
  constexpr int SECONDS_PER_DAY = 86400;
  constexpr int DAYS_PER_WEEK = 7;
  constexpr int MONTHS_PER_YEAR = 12;
  constexpr int DAYS_PER_NON_LEAP_YEAR = 365;
  constexpr int DAYS_PER_LEAP_YEAR = 366;
  constexpr int DAYS_PER4_YEARS = (3 * DAYS_PER_NON_LEAP_YEAR + DAYS_PER_LEAP_YEAR);
  constexpr int DAYS_PER100_YEARS = (25 * DAYS_PER4_YEARS - 1);
  constexpr int DAYS_PER400_YEARS = (4 * DAYS_PER100_YEARS + 1);
  constexpr int TIME_YEAR_BASE = 1900;
  constexpr int EPOCH_YEAR = 1970;
  constexpr int WEEK_DAY_OF2000_MARCH_FIRST = 3;
  constexpr int64_t SECONDS_UNTIL2000_MARCH_FIRST = 951868800;
  constexpr int NON_LEAP_YEAR_DAYS_IN_MONTH[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
}

inline bool is_leap_year(const int64_t year) {
  return (((year) % 4) == 0 && (((year) % 100) != 0 || ((year) % 400) == 0));
}

// OLD ALGORITHM
static int64_t computeRemainingYears(int64_t daysPerYears,
                                     int64_t quotientYears,
                                     int64_t *remainingDays) {
  int64_t years = *remainingDays / daysPerYears;
  if (years == quotientYears)
    years--;
  *remainingDays -= years * daysPerYears;
  return years;
}

int64_t update_from_seconds_old(time_t total_seconds, struct tm *tm) {
  static const char daysInMonth[] = {31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29};

  int64_t seconds = total_seconds - time_constants::SECONDS_UNTIL2000_MARCH_FIRST;
  int64_t days = seconds / time_constants::SECONDS_PER_DAY;
  int64_t remainingSeconds = seconds % time_constants::SECONDS_PER_DAY;
  if (remainingSeconds < 0) {
    remainingSeconds += time_constants::SECONDS_PER_DAY;
    days--;
  }

  int64_t wday = (time_constants::WEEK_DAY_OF2000_MARCH_FIRST + days) %
                 time_constants::DAYS_PER_WEEK;
  if (wday < 0)
    wday += time_constants::DAYS_PER_WEEK;

  int64_t numOfFourHundredYearCycles = days / time_constants::DAYS_PER400_YEARS;
  int64_t remainingDays = days % time_constants::DAYS_PER400_YEARS;
  if (remainingDays < 0) {
    remainingDays += time_constants::DAYS_PER400_YEARS;
    numOfFourHundredYearCycles--;
  }

  int64_t numOfHundredYearCycles = computeRemainingYears(
      time_constants::DAYS_PER100_YEARS, 4, &remainingDays);
  int64_t numOfFourYearCycles = computeRemainingYears(
      time_constants::DAYS_PER4_YEARS, 25, &remainingDays);
  int64_t remainingYears = computeRemainingYears(
      time_constants::DAYS_PER_NON_LEAP_YEAR, 4, &remainingDays);

  int64_t years = remainingYears + 4 * numOfFourYearCycles +
                  100 * numOfHundredYearCycles +
                  400LL * numOfFourHundredYearCycles;

  int leapDay =
      !remainingYears && (numOfFourYearCycles || !numOfHundredYearCycles);

  int64_t yday = remainingDays + 31 + 28 + leapDay;
  if (yday >= time_constants::DAYS_PER_NON_LEAP_YEAR + leapDay)
    yday -= time_constants::DAYS_PER_NON_LEAP_YEAR + leapDay;

  int64_t months = 0;
  while (daysInMonth[months] <= remainingDays) {
    remainingDays -= daysInMonth[months];
    months++;
  }

  if (months >= time_constants::MONTHS_PER_YEAR - 2) {
    months -= time_constants::MONTHS_PER_YEAR;
    years++;
  }

  tm->tm_year = static_cast<int>(years + 2000 - time_constants::TIME_YEAR_BASE);
  tm->tm_mon = static_cast<int>(months + 2);
  tm->tm_mday = static_cast<int>(remainingDays + 1);
  tm->tm_wday = static_cast<int>(wday);
  tm->tm_yday = static_cast<int>(yday);
  tm->tm_hour =
      static_cast<int>(remainingSeconds / time_constants::SECONDS_PER_HOUR);
  tm->tm_min =
      static_cast<int>(remainingSeconds / time_constants::SECONDS_PER_MIN %
                       time_constants::SECONDS_PER_MIN);
  tm->tm_sec =
      static_cast<int>(remainingSeconds % time_constants::SECONDS_PER_MIN);
  tm->tm_isdst = 0;

  return 0;
}

// NEW FAST ALGORITHM
int64_t update_from_seconds_fast(time_t total_seconds, struct tm *tm) {
  int64_t days = total_seconds / time_constants::SECONDS_PER_DAY;
  int64_t remaining_seconds = total_seconds % time_constants::SECONDS_PER_DAY;
  if (remaining_seconds < 0) {
    remaining_seconds += time_constants::SECONDS_PER_DAY;
    days--;
  }

  days += 719528;
  days -= 60;

  const int64_t era = (days >= 0 ? days : days - 146096) / 146097;
  const int64_t doe = days - era * 146097;
  const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int y = static_cast<int>(yoe + era * 400);
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int64_t mp = (5 * doy + 2) / 153;
  const int d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);

  const int month = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
  const int year = y + (mp >= 10);

  const bool is_leap = (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
  int yday;
  if (mp < 10) {
    yday = static_cast<int>(doy + (is_leap ? 60 : 59));
  } else {
    yday = static_cast<int>(doy - 306);
  }

  const int64_t unix_days = total_seconds / time_constants::SECONDS_PER_DAY;
  int wday = static_cast<int>((unix_days + 4) % 7);
  if (wday < 0)
    wday += 7;

  tm->tm_year = year - time_constants::TIME_YEAR_BASE;
  tm->tm_mon = month - 1;
  tm->tm_mday = d;
  tm->tm_wday = wday;
  tm->tm_yday = yday;
  tm->tm_hour = static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_HOUR);
  tm->tm_min = static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_MIN %
                       time_constants::SECONDS_PER_MIN);
  tm->tm_sec = static_cast<int>(remaining_seconds % time_constants::SECONDS_PER_MIN);
  tm->tm_isdst = 0;

  return 0;
}

// Benchmark helper
template<typename Func>
double benchmark(const char* name, Func func, int iterations) {
  printf("Running %s (%d iterations)...\n", name, iterations);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < iterations; i++) {
    struct tm result;
    time_t ts = i * 1000LL;  // Spread timestamps across range
    func(ts, &result);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  
  double time_sec = elapsed.count();
  double ns_per_op = (time_sec * 1e9) / iterations;
  double ops_per_sec = iterations / time_sec;
  
  printf("  Time: %.3f seconds\n", time_sec);
  printf("  Rate: %.2f million ops/sec\n", ops_per_sec / 1e6);
  printf("  Avg:  %.2f ns per conversion\n\n", ns_per_op);
  
  return time_sec;
}

int main() {
  printf("========================================\n");
  printf("Phase 4: Performance Benchmark\n");
  printf("========================================\n\n");
  
  const int ITERATIONS = 10000000;  // 10 million
  
  // Warm-up
  printf("Warming up...\n");
  for (int i = 0; i < 1000; i++) {
    struct tm result;
    update_from_seconds_old(i * 1000LL, &result);
    update_from_seconds_fast(i * 1000LL, &result);
  }
  printf("\n");
  
  // Benchmark old algorithm
  double time_old = benchmark("Old Algorithm", update_from_seconds_old, ITERATIONS);
  
  // Benchmark fast algorithm
  double time_fast = benchmark("Fast Algorithm", update_from_seconds_fast, ITERATIONS);
  
  // Results
  printf("========================================\n");
  printf("Benchmark Results\n");
  printf("========================================\n");
  printf("Old algorithm:  %.3f seconds\n", time_old);
  printf("Fast algorithm: %.3f seconds\n", time_fast);
  
  if (time_old > 0 && time_fast > 0) {
    double speedup_pct = ((time_old - time_fast) / time_old) * 100.0;
    double ratio = time_old / time_fast;
    
    printf("\n");
    if (speedup_pct > 0) {
      printf("✓ Fast algorithm is %.1f%% faster\n", speedup_pct);
      printf("  (%.2fx speedup)\n", ratio);
    } else {
      printf("⚠ Fast algorithm is %.1f%% slower\n", -speedup_pct);
      printf("  (%.2fx slowdown)\n", 1.0/ratio);
    }
  }
  
  // Sequential dates benchmark (better cache locality)
  printf("\n========================================\n");
  printf("Sequential Dates Benchmark\n");
  printf("========================================\n\n");
  
  auto seq_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ITERATIONS; i++) {
    struct tm result;
    update_from_seconds_old(i * 86400LL, &result);  // One per day
  }
  auto seq_mid = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < ITERATIONS; i++) {
    struct tm result;
    update_from_seconds_fast(i * 86400LL, &result);
  }
  auto seq_end = std::chrono::high_resolution_clock::now();
  
  double seq_time_old = std::chrono::duration<double>(seq_mid - seq_start).count();
  double seq_time_fast = std::chrono::duration<double>(seq_end - seq_mid).count();
  
  printf("Old algorithm:  %.3f seconds\n", seq_time_old);
  printf("Fast algorithm: %.3f seconds\n", seq_time_fast);
  
  if (seq_time_old > 0 && seq_time_fast > 0) {
    double seq_speedup = ((seq_time_old - seq_time_fast) / seq_time_old) * 100.0;
    printf("Speedup: %.1f%%\n", seq_speedup);
  }
  
  printf("\n✓ Benchmark complete\n");
  return 0;
}
