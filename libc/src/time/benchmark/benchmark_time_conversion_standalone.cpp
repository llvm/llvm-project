//===-- Standalone benchmark for time conversion functions ----------------===//
//
// Compares performance of update_from_seconds_fast vs unix_to_date_fast
// This is a standalone version with all code inline for easy compilation
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <climits>

using namespace std::chrono;

// ============================================================================
// Constants from time_constants.h
// ============================================================================
namespace time_constants {
constexpr int64_t SECONDS_PER_DAY = 86400;
constexpr int64_t SECONDS_PER_HOUR = 3600;
constexpr int64_t SECONDS_PER_MIN = 60;
constexpr int64_t DAYS_PER_WEEK = 7;
constexpr int64_t TIME_YEAR_BASE = 1900;
constexpr int64_t NUMBER_OF_SECONDS_IN_LEAP_YEAR = 31622400;
}

// ============================================================================
// Implementation 1: update_from_seconds_fast (Howard Hinnant style)
// ============================================================================
int64_t update_from_seconds_fast(time_t total_seconds, struct tm *tm) {
  constexpr time_t time_min =
      (sizeof(time_t) == 4)
          ? INT_MIN
          : INT_MIN * static_cast<int64_t>(
                          time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR);
  constexpr time_t time_max =
      (sizeof(time_t) == 4)
          ? INT_MAX
          : INT_MAX * static_cast<int64_t>(
                          time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR);

  if (total_seconds < time_min || total_seconds > time_max)
    return -1;

  int64_t days = total_seconds / time_constants::SECONDS_PER_DAY;
  int64_t remaining_seconds = total_seconds % time_constants::SECONDS_PER_DAY;
  if (remaining_seconds < 0) {
    remaining_seconds += time_constants::SECONDS_PER_DAY;
    days--;
  }

  days += 719528;  // Convert to days since 0000-01-01
  days -= 60;      // Shift to March-based year

  const int64_t era = (days >= 0 ? days : days - 146096) / 146097;
  const int64_t doe = days - era * 146097;
  const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int y = static_cast<int>(yoe + era * 400);
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int64_t mp = (5 * doy + 2) / 153;
  const int d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
  const int month = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
  const int year = y + (mp >= 10);

  if (year > INT_MAX || year < INT_MIN)
    return -1;

  const bool is_leap =
      (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
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
  tm->tm_hour =
      static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_HOUR);
  tm->tm_min =
      static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_MIN %
                       time_constants::SECONDS_PER_MIN);
  tm->tm_sec =
      static_cast<int>(remaining_seconds % time_constants::SECONDS_PER_MIN);
  tm->tm_isdst = 0;

  return 0;
}

// ============================================================================
// Implementation 2: unix_to_date_fast (Ben Joffe style)
// ============================================================================
namespace fast_date {

struct DateResult {
  int year;
  int month;
  int day;
  int yday;
  int wday;
  int hour;
  int minute;
  int second;
  bool valid;
};

constexpr int64_t SECONDS_PER_DAY = 86400;
constexpr int64_t SECONDS_PER_HOUR = 3600;
constexpr int64_t SECONDS_PER_MINUTE = 60;
constexpr int64_t UNIX_EPOCH_DAYS = 719528;
constexpr int64_t MARCH_SHIFT_DAYS = 60;
constexpr int64_t DAYS_PER_ERA = 146097;
constexpr int64_t DAYS_PER_CENTURY = 36524;
constexpr int64_t DAYS_PER_4_YEARS = 1461;
constexpr int64_t YEARS_PER_ERA = 400;
constexpr int64_t YEARS_PER_CENTURY = 100;
constexpr int64_t MONTH_CYCLE_DAYS = 153;
constexpr int64_t MONTH_CYCLE_MONTHS = 5;
constexpr int UNIX_EPOCH_WDAY = 4;
constexpr int DAYS_BEFORE_MARCH = 306;

void days_to_ymd_joffe(int64_t days, int &year, int &month, int &day, int &yday) {
  days -= MARCH_SHIFT_DAYS;
  
  const int64_t era = (days >= 0 ? days : days - (DAYS_PER_ERA - 1)) / DAYS_PER_ERA;
  const int64_t doe = days - era * DAYS_PER_ERA;
  const int64_t yoe = (doe - doe / DAYS_PER_4_YEARS + doe / DAYS_PER_CENTURY - doe / DAYS_PER_ERA) / 365;
  const int y = static_cast<int>(yoe + era * YEARS_PER_ERA);
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / YEARS_PER_CENTURY);
  const int64_t mp = (MONTH_CYCLE_MONTHS * doy + 2) / MONTH_CYCLE_DAYS;
  const int d = static_cast<int>(doy - (MONTH_CYCLE_DAYS * mp + 2) / MONTH_CYCLE_MONTHS + 1);
  
  month = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
  year = y + (mp >= 10);
  day = d;
  
  const bool is_leap = (year % 4 == 0) && ((year % YEARS_PER_CENTURY != 0) || (year % YEARS_PER_ERA == 0));
  if (mp < 10) {
    yday = static_cast<int>(doy + (is_leap ? MARCH_SHIFT_DAYS : MARCH_SHIFT_DAYS - 1));
  } else {
    yday = static_cast<int>(doy - DAYS_BEFORE_MARCH);
  }
}

DateResult unix_to_date_fast(int64_t timestamp) {
  DateResult result = {0, 0, 0, 0, 0, 0, 0, 0, false};
  
  int64_t days = timestamp / SECONDS_PER_DAY;
  int64_t remaining = timestamp % SECONDS_PER_DAY;
  
  if (remaining < 0) {
    remaining += SECONDS_PER_DAY;
    days--;
  }
  
  days += UNIX_EPOCH_DAYS;
  
  days_to_ymd_joffe(days, result.year, result.month, result.day, result.yday);
  
  result.hour = static_cast<int>(remaining / SECONDS_PER_HOUR);
  remaining %= SECONDS_PER_HOUR;
  result.minute = static_cast<int>(remaining / SECONDS_PER_MINUTE);
  result.second = static_cast<int>(remaining % SECONDS_PER_MINUTE);
  
  int64_t total_days = timestamp / SECONDS_PER_DAY;
  result.wday = static_cast<int>((total_days + UNIX_EPOCH_WDAY) % 7);
  if (result.wday < 0) result.wday += 7;
  
  result.valid = true;
  return result;
}

} // namespace fast_date

// ============================================================================
// Benchmark code
// ============================================================================

constexpr int WARMUP_ITERATIONS = 10000;
constexpr int BENCHMARK_ITERATIONS = 1000000;

std::vector<time_t> generate_test_timestamps() {
    std::vector<time_t> timestamps;
    
    timestamps.push_back(0);
    timestamps.push_back(946684800);
    timestamps.push_back(1000000000);
    timestamps.push_back(1234567890);
    timestamps.push_back(1500000000);
    timestamps.push_back(1700000000);
    timestamps.push_back(2000000000);
    timestamps.push_back(2147483647);
    timestamps.push_back(951868800);
    timestamps.push_back(1077926400);
    timestamps.push_back(1078012800);
    timestamps.push_back(1235865600);
    timestamps.push_back(946684799);
    timestamps.push_back(946684800);
    timestamps.push_back(1609459199);
    timestamps.push_back(1609459200);
    timestamps.push_back(-86400);
    timestamps.push_back(-946684800);
    timestamps.push_back(-2208988800);
    
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<time_t> dist(-2208988800, 2147483647);
    for (int i = 0; i < 50; i++) {
        timestamps.push_back(dist(gen));
    }
    
    return timestamps;
}

double benchmark_update_from_seconds_fast(const std::vector<time_t>& timestamps, int iterations) {
    struct tm result;
    volatile int64_t return_code = 0;
    
    auto start = high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (time_t ts : timestamps) {
            return_code = update_from_seconds_fast(ts, &result);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    // Use return_code to prevent optimization
    if (return_code < -1000000) std::cout << "";
    
    return static_cast<double>(duration) / (iterations * timestamps.size());
}

double benchmark_unix_to_date_fast(const std::vector<time_t>& timestamps, int iterations) {
    fast_date::DateResult result;
    
    auto start = high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (time_t ts : timestamps) {
            result = fast_date::unix_to_date_fast(ts);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    // Use result to prevent optimization
    if (!result.valid && result.year < -1000000) std::cout << "";
    
    return static_cast<double>(duration) / (iterations * timestamps.size());
}

bool verify_correctness(const std::vector<time_t>& timestamps) {
    int mismatches = 0;
    bool all_correct = true;
    
    for (time_t ts : timestamps) {
        struct tm tm_result;
        std::memset(&tm_result, 0, sizeof(struct tm));
        int64_t ret1 = update_from_seconds_fast(ts, &tm_result);
        
        fast_date::DateResult fast_result = fast_date::unix_to_date_fast(ts);
        
        bool match = true;
        if (ret1 == 0 && fast_result.valid) {
            if (tm_result.tm_year != fast_result.year - 1900 ||
                tm_result.tm_mon != fast_result.month - 1 ||
                tm_result.tm_mday != fast_result.day ||
                tm_result.tm_hour != fast_result.hour ||
                tm_result.tm_min != fast_result.minute ||
                tm_result.tm_sec != fast_result.second ||
                tm_result.tm_wday != fast_result.wday ||
                tm_result.tm_yday != fast_result.yday) {
                match = false;
            }
        } else if ((ret1 != 0 && fast_result.valid) || (ret1 == 0 && !fast_result.valid)) {
            match = false;
        }
        
        if (!match) {
            mismatches++;
            all_correct = false;
            if (mismatches <= 5) {
                std::cout << "Mismatch for timestamp " << ts << ":\n";
                std::cout << "  update_from_seconds_fast: " 
                          << (ret1 == 0 ? "success" : "error") << "\n";
                if (ret1 == 0) {
                    std::cout << "    " << (1900 + tm_result.tm_year) << "-" 
                              << std::setfill('0') << std::setw(2) << (tm_result.tm_mon + 1) << "-"
                              << std::setw(2) << tm_result.tm_mday << " "
                              << std::setw(2) << tm_result.tm_hour << ":"
                              << std::setw(2) << tm_result.tm_min << ":"
                              << std::setw(2) << tm_result.tm_sec 
                              << " (wday=" << tm_result.tm_wday << ", yday=" << tm_result.tm_yday << ")\n";
                }
                std::cout << "  unix_to_date_fast: " 
                          << (fast_result.valid ? "success" : "error") << "\n";
                if (fast_result.valid) {
                    std::cout << "    " << fast_result.year << "-" 
                              << std::setfill('0') << std::setw(2) << fast_result.month << "-"
                              << std::setw(2) << fast_result.day << " "
                              << std::setw(2) << fast_result.hour << ":"
                              << std::setw(2) << fast_result.minute << ":"
                              << std::setw(2) << fast_result.second
                              << " (wday=" << fast_result.wday << ", yday=" << fast_result.yday << ")\n";
                }
            }
        }
    }
    
    if (mismatches > 0) {
        std::cout << "\nTotal mismatches: " << mismatches << " out of " 
                  << timestamps.size() << " timestamps\n";
    }
    
    return all_correct;
}

int main() {
    std::cout << "=== Time Conversion Benchmark (Standalone) ===\n\n";
    
    std::vector<time_t> timestamps = generate_test_timestamps();
    std::cout << "Generated " << timestamps.size() << " test timestamps\n\n";
    
    std::cout << "Verifying correctness...\n";
    bool correct = verify_correctness(timestamps);
    if (correct) {
        std::cout << "✓ All results match!\n\n";
    } else {
        std::cout << "✗ Results differ - see details above\n\n";
    }
    
    std::cout << "Warming up (" << WARMUP_ITERATIONS << " iterations)...\n";
    benchmark_update_from_seconds_fast(timestamps, WARMUP_ITERATIONS);
    benchmark_unix_to_date_fast(timestamps, WARMUP_ITERATIONS);
    std::cout << "Warmup complete\n\n";
    
    std::cout << "Running benchmarks (" << BENCHMARK_ITERATIONS << " iterations)...\n\n";
    
    double time1 = benchmark_update_from_seconds_fast(timestamps, BENCHMARK_ITERATIONS);
    std::cout << "update_from_seconds_fast: " << std::fixed << std::setprecision(2) 
              << time1 << " ns/conversion\n";
    
    double time2 = benchmark_unix_to_date_fast(timestamps, BENCHMARK_ITERATIONS);
    std::cout << "unix_to_date_fast:        " << std::fixed << std::setprecision(2) 
              << time2 << " ns/conversion\n\n";
    
    double speedup = time1 / time2;
    double improvement = ((time1 - time2) / time1) * 100.0;
    
    std::cout << "=== Results ===\n";
    if (speedup > 1.0) {
        std::cout << "unix_to_date_fast is " << std::fixed << std::setprecision(2) 
                  << speedup << "x FASTER (" 
                  << std::setprecision(1) << improvement << "% improvement)\n";
    } else {
        std::cout << "update_from_seconds_fast is " << std::fixed << std::setprecision(2) 
                  << (1.0 / speedup) << "x FASTER (" 
                  << std::setprecision(1) << -improvement << "% improvement)\n";
    }
    
    return correct ? 0 : 1;
}
