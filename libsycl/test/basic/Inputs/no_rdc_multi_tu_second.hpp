#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>

void runSecondTuKernel(sycl::queue &Q, int *Data, std::size_t N);
