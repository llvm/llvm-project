#ifndef BENCHMARK_LOG_H_
#define BENCHMARK_LOG_H_

#include <ostream>

namespace benchmark {
namespace internal {

int GetLogLevel();
void SetLogLevel(int level);

std::ostream& GetNullLogInstance();
std::ostream& GetErrorLogInstance();

inline std::ostream& GetLogInstanceForLevel(int level) {
  if (level <= GetLogLevel()) {
    return GetErrorLogInstance();
  }
  return GetNullLogInstance();
}

} // end namespace internal
} // end namespace benchmark

#define VLOG(x) (::benchmark::internal::GetLogInstanceForLevel(x) \
                 << "-- LOG(" << x << "): ")

#endif