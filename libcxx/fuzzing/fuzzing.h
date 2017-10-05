// -*- C++ -*-
//===-------------------------- fuzzing.h --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_FUZZING
#define _LIBCPP_FUZZING

#include <cstddef> // for size_t
#include <cstdint> // for uint8_t

namespace fuzzing {

//	These all return 0 on success; != 0 on failure
	int sort             (const uint8_t *data, size_t size);
	int stable_sort      (const uint8_t *data, size_t size);
	int partition        (const uint8_t *data, size_t size);
	int stable_partition (const uint8_t *data, size_t size);

//	partition and stable_partition take Bi-Di iterators.
//	Should test those, too

	int nth_element      (const uint8_t *data, size_t size);
	int partial_sort     (const uint8_t *data, size_t size);
	
} // namespace fuzzing

#endif // _LIBCPP_FUZZING
