#include <chrono>

using namespace std::chrono;

constexpr bool test_month() {
  using decamonths = duration<int, std::ratio_multiply<std::ratio<10>, months::period>>;
  auto ymd = 2001y/January/1d;
  auto ymd2 = 2001y/January/1d;
  ymd2 += std::chrono::duration_cast<std::chrono::months>(decamonths(1));
  ymd += decamonths(1);


  if (ymd2.month() != November)
    return false;
  
  return ymd == 2001y/November/1d;
}

constexpr bool test_year() {
  using decades = duration<int, std::ratio_multiply<std::ratio<10>, years::period>>;
  auto ymd = 2001y/January/1d;
  
  ymd += duration_cast<years>(decades(1)); 
  
  return ymd == 2011y/January/1d;
}

int main(int, char**) {
  static_assert(test_month(), "Month arithmetic failed");
  static_assert(test_year(), "Year arithmetic failed");

  return 0;
}
