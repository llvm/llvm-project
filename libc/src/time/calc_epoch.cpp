#include <iostream>

int64_t civil_from_days_test(int64_t z) {
    z -= 719468;  // Hinnant uses 1970-01-01 = 719468 days from 0000-03-01
    return z;
}

int main() {
  // Hinnant's algorithm uses 1970-01-01 as 719468 days from March 1, year 0
  // So: 1970-01-01 from 0000-03-01 = 719468
  // 0000-03-01 is 60 days after 0000-01-01 (31 Jan + 29 Feb, year 0 is leap)
  // So: 1970-01-01 from 0000-01-01 = 719468 + 60 = 719528
  std::cout << "1970-01-01 from 0000-01-01 should be: 719528\n";
  std::cout << "1970-01-01 from 0000-03-01 is: 719468\n";
  std::cout << "Difference (days in Jan+Feb year 0): " << (719528 - 719468) << "\n";
  
  return 0;
}
