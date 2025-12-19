#include "fast_date.h"
#include <iostream>
#include <ctime>

using namespace fast_date;

int main() {
  // 2000-02-29 00:00:00 UTC
  int64_t ts = 951868800;
  DateResult result = unix_to_date_fast(ts);
  std::cout << "Timestamp: " << ts << "\n";
  std::cout << "Result: " << result.year << "-" << result.month << "-" << result.day << "\n";
  
  // Verify with system
  time_t t = ts;
  struct tm *gmt = gmtime(&t);
  std::cout << "System: " << (1900 + gmt->tm_year) << "-" << (gmt->tm_mon + 1) << "-" << gmt->tm_mday << "\n";
  
  // Also test 2400-02-29
  ts = 13574476800;
  result = unix_to_date_fast(ts);
  std::cout << "\nTimestamp: " << ts << "\n";
  std::cout << "Result: " << result.year << "-" << result.month << "-" << result.day << "\n";
  t = ts;
  gmt = gmtime(&t);
  std::cout << "System: " << (1900 + gmt->tm_year) << "-" << (gmt->tm_mon + 1) << "-" << gmt->tm_mday << "\n";
}
