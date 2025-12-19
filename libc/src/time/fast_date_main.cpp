//===-- Test program for fast date algorithm -----------------------------===//
//
// Simple test program to demonstrate the fast date conversion
//
//===----------------------------------------------------------------------===//

#include "fast_date.h"
#include <cstdio>
#include <ctime>
#include <cstring>

using namespace fast_date;

// Helper to print a date result
void print_date(const DateResult &date) {
  const char* weekdays[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
  const char* months[] = {"", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  
  if (!date.valid) {
    printf("Invalid date\n");
    return;
  }
  
  printf("%s %s %02d %02d:%02d:%02d %04d (yday=%d)\n",
         weekdays[date.wday],
         months[date.month],
         date.day,
         date.hour, date.minute, date.second,
         date.year,
         date.yday);
}

// Compare with system gmtime for validation
void compare_with_system(int64_t timestamp) {
  // Our implementation
  DateResult fast = unix_to_date_fast(timestamp);
  
  // System implementation
  time_t t = static_cast<time_t>(timestamp);
  struct tm* sys = gmtime(&t);
  
  printf("\nTimestamp: %lld\n", (long long)timestamp);
  printf("Fast:   ");
  print_date(fast);
  
  if (sys) {
    printf("System: %s %s %02d %02d:%02d:%02d %04d (yday=%d)\n",
           (const char*[]){"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"}[sys->tm_wday],
           (const char*[]){"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}[sys->tm_mon],
           sys->tm_mday,
           sys->tm_hour, sys->tm_min, sys->tm_sec,
           sys->tm_year + 1900,
           sys->tm_yday);
    
    // Check if they match
    bool matches = (fast.year == sys->tm_year + 1900) &&
                   (fast.month == sys->tm_mon + 1) &&
                   (fast.day == sys->tm_mday) &&
                   (fast.hour == sys->tm_hour) &&
                   (fast.minute == sys->tm_min) &&
                   (fast.second == sys->tm_sec) &&
                   (fast.wday == sys->tm_wday) &&
                   (fast.yday == sys->tm_yday);
    
    if (matches) {
      printf("✓ MATCH\n");
    } else {
      printf("✗ MISMATCH!\n");
    }
  } else {
    printf("System: (gmtime failed)\n");
  }
}

// Test the inverse function
void test_inverse(int year, int month, int day) {
  printf("\nTesting inverse: %04d-%02d-%02d\n", year, month, day);
  
  // Convert to timestamp
  int64_t timestamp = date_to_unix_fast(year, month, day, 12, 30, 45);
  printf("Timestamp: %lld\n", (long long)timestamp);
  
  // Convert back to date
  DateResult result = unix_to_date_fast(timestamp);
  printf("Round-trip: ");
  print_date(result);
  
  // Check if it matches
  if (result.year == year && result.month == month && result.day == day) {
    printf("✓ Round-trip successful\n");
  } else {
    printf("✗ Round-trip failed!\n");
  }
}

int main() {
  printf("===========================================\n");
  printf("Fast Date Algorithm Test (Joffe Algorithm)\n");
  printf("===========================================\n\n");
  
  printf("Testing key dates:\n");
  printf("------------------\n");
  
  // Unix epoch
  compare_with_system(0);
  
  // Y2K
  compare_with_system(946684800); // 2000-01-01 00:00:00
  
  // Leap year date
  compare_with_system(951868800); // 2000-02-29 00:00:00
  
  // Current time (approximate)
  compare_with_system(1700000000); // 2023-11-14 22:13:20
  
  // Future date
  compare_with_system(2147483647); // 2038-01-19 03:14:07 (32-bit limit)
  
  // Negative timestamp (before epoch)
  compare_with_system(-86400); // 1969-12-31 00:00:00
  
  // Far past
  compare_with_system(-2208988800); // 1900-01-01 00:00:00
  
  printf("\n\nTesting inverse function:\n");
  printf("-------------------------\n");
  
  test_inverse(2000, 1, 1);
  test_inverse(2000, 2, 29);  // Leap day
  test_inverse(2024, 12, 25); // Christmas 2024
  test_inverse(1970, 1, 1);   // Unix epoch
  test_inverse(2038, 1, 19);  // 32-bit limit
  
  printf("\n\nPerformance test:\n");
  printf("-----------------\n");
  
  // Simple performance test
  const int64_t iterations = 10000000;
  int64_t start_ts = 0;
  
  printf("Converting %lld timestamps...\n", (long long)iterations);
  
  clock_t start = clock();
  for (int64_t i = 0; i < iterations; i++) {
    DateResult r = unix_to_date_fast(start_ts + i * 86400);
    // Prevent optimization from removing the loop
    start_ts += (r.year & 1);
  }
  clock_t end = clock();
  
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
  double per_conversion = (elapsed / iterations) * 1e9; // nanoseconds
  
  printf("Time: %.3f seconds\n", elapsed);
  printf("Rate: %.2f million conversions/sec\n", iterations / elapsed / 1e6);
  printf("Avg:  %.2f ns per conversion\n", per_conversion);
  
  return 0;
}
