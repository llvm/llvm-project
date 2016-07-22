#ifndef BENCHMARK_WALLTIME_H_
#define BENCHMARK_WALLTIME_H_

#include <string>

namespace benchmark {
typedef double WallTime;

namespace walltime {
WallTime Now();
}  // end namespace walltime

std::string LocalDateTimeString();

}  // end namespace benchmark

#endif  // BENCHMARK_WALLTIME_H_
