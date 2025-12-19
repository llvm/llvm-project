#include "time_utils.h"
#include <cstdio>
#include <ctime>
#include <cstring>

using namespace LIBC_NAMESPACE;

// Helper to format tm as string
const char* format_tm(const tm* t) {
  static char buf[100];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d wday=%d yday=%d",
           t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
           t->tm_hour, t->tm_min, t->tm_sec,
           t->tm_wday, t->tm_yday);
  return buf;
}

bool compare_tm(const tm* a, const tm* b) {
  return a->tm_year == b->tm_year &&
         a->tm_mon == b->tm_mon &&
         a->tm_mday == b->tm_mday &&
         a->tm_hour == b->tm_hour &&
         a->tm_min == b->tm_min &&
         a->tm_sec == b->tm_sec &&
         a->tm_wday == b->tm_wday &&
         a->tm_yday == b->tm_yday;
}

void test_timestamp(time_t ts, const char* description) {
  tm result_old, result_fast;
  memset(&result_old, 0, sizeof(tm));
  memset(&result_fast, 0, sizeof(tm));
  
  int64_t ret_old = time_utils::update_from_seconds(ts, &result_old);
  int64_t ret_fast = time_utils::update_from_seconds_fast(ts, &result_fast);
  
  printf("\n%s (ts=%ld):\n", description, ts);
  printf("  Old:  %s (ret=%ld)\n", format_tm(&result_old), ret_old);
  printf("  Fast: %s (ret=%ld)\n", format_tm(&result_fast), ret_fast);
  
  if (ret_old == ret_fast && compare_tm(&result_old, &result_fast)) {
    printf("  ✓ MATCH\n");
  } else {
    printf("  ✗ MISMATCH!\n");
  }
}

int main() {
  printf("========================================\n");
  printf("Integration Test: Old vs Fast Algorithm\n");
  printf("========================================\n");
  
  // Test key dates
  test_timestamp(0, "Unix epoch (1970-01-01)");
  test_timestamp(946684800, "Y2K (2000-01-01)");
  test_timestamp(951782400, "Leap day 2000 (2000-02-29)");
  test_timestamp(1700000000, "Recent date (2023-11-14)");
  test_timestamp(2147483647, "32-bit max (2038-01-19)");
  test_timestamp(-86400, "Before epoch (1969-12-31)");
  test_timestamp(-2208988800, "Year 1900 (1900-01-01)");
  test_timestamp(13574563200, "Far future (2400-02-29)");
  
  // Test all months of 2024
  printf("\n\nTesting all months of 2024:\n");
  for (int month = 1; month <= 12; month++) {
    time_t ts;
    if (month == 1) ts = 1704067200; // 2024-01-01
    else if (month == 2) ts = 1706745600; // 2024-02-01
    else if (month == 3) ts = 1709251200; // 2024-03-01
    else if (month == 4) ts = 1711929600; // 2024-04-01
    else if (month == 5) ts = 1714521600; // 2024-05-01
    else if (month == 6) ts = 1717200000; // 2024-06-01
    else if (month == 7) ts = 1719792000; // 2024-07-01
    else if (month == 8) ts = 1722470400; // 2024-08-01
    else if (month == 9) ts = 1725148800; // 2024-09-01
    else if (month == 10) ts = 1727740800; // 2024-10-01
    else if (month == 11) ts = 1730419200; // 2024-11-01
    else ts = 1733011200; // 2024-12-01
    
    char desc[50];
    snprintf(desc, sizeof(desc), "2024-%02d-01", month);
    test_timestamp(ts, desc);
  }
  
  // Performance test
  printf("\n\nPerformance test (10M conversions):\n");
  const int N = 10000000;
  
  time_t start = time(nullptr);
  for (int i = 0; i < N; i++) {
    tm result;
    time_utils::update_from_seconds(i * 1000, &result);
  }
  time_t end1 = time(nullptr);
  
  for (int i = 0; i < N; i++) {
    tm result;
    time_utils::update_from_seconds_fast(i * 1000, &result);
  }
  time_t end2 = time(nullptr);
  
  double time_old = (end1 - start);
  double time_fast = (end2 - end1);
  
  printf("  Old algorithm:  %.3f seconds\n", time_old);
  printf("  Fast algorithm: %.3f seconds\n", time_fast);
  if (time_old > 0 && time_fast > 0) {
    double speedup = ((time_old - time_fast) / time_old) * 100.0;
    printf("  Speedup: %.1f%%\n", speedup);
  }
  
  printf("\n✓ Integration test complete\n");
  return 0;
}
