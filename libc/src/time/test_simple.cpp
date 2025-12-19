#include <iostream>
int main() {
  // Test the month calculation for day 0 (epoch)
  int64_t days = 719468;  // Unix epoch in days since 0000-01-01
  days += 306;
  int64_t qday = days * 4 + 3;
  int64_t cent = qday / 146097;
  int64_t qjul = qday - (cent & ~3) + cent * 4;
  int year = qjul / 1461;
  int64_t yday_march = (qjul % 1461) / 4;
  
  int64_t N = yday_march * 2141 + 197913;
  int M = N / 65536;
  int D = (N % 65536) / 2141;
  
  bool bump = (yday_march >= 306);
  
  std::cout << "days=" << (days-306) << " yday_march=" << yday_march << " M=" << M << " D=" << D << " bump=" << bump << "\n";
  std::cout << "year=" << year << " month=" << (bump ? M - 9 : M + 3) << " day=" << (D+1) << "\n";
  
  return 0;
}
