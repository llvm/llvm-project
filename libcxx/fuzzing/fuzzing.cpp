// -*- C++ -*-
//===------------------------- fuzzing.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//	A set of routines to use when fuzzing the algorithms in libc++
//	Each one tests a single algorithm.
//
//	They all have the form of:
//		int `algorithm`(const uint8_t *data, size_t size);
//
//	They perform the operation, and then check to see if the results are correct.
//	If so, they return zero, and non-zero otherwise.
//
//	For example, sort calls std::sort, then checks two things:
//		(1) The resulting vector is sorted
//		(2) The resulting vector contains the same elements as the original data.



#include "fuzzing.h"
#include <vector>
#include <algorithm>
#include <regex>

//	If we had C++14, we could use the four iterator version of is_permutation

namespace fuzzing {

//	This is a struct we can use to test the stable_XXX algorithms.
//	perform the operation on the key, then check the order of the payload.

struct stable_test {
	uint8_t key;
	uint8_t payload;
	
	stable_test(uint8_t k) : key(k), payload(0) {}
	stable_test(uint8_t k, uint8_t p) : key(k), payload(p) {}
	};

void swap(stable_test &lhs, stable_test &rhs)
{
	using std::swap;
	swap(lhs.key,     rhs.key);
	swap(lhs.payload, rhs.payload);
}

struct key_less
{
	bool operator () (const stable_test &lhs, const stable_test &rhs) const
	{
		return lhs.key < rhs.key;
	}
};

struct payload_less
{
	bool operator () (const stable_test &lhs, const stable_test &rhs) const
	{
		return lhs.payload < rhs.payload;
	}
};

struct total_less
{
	bool operator () (const stable_test &lhs, const stable_test &rhs) const
	{
		return lhs.key == rhs.key ? lhs.payload < rhs.payload : lhs.key < rhs.key;
	}
};

bool operator==(const stable_test &lhs, const stable_test &rhs)
{ 
	return lhs.key == rhs.key && lhs.payload == rhs.payload;
}


template<typename T>
struct is_even
{
	bool operator () (const T &t) const
	{
		return t % 2 == 0;
	}
};


template<>
struct is_even<stable_test>
{
	bool operator () (const stable_test &t) const
	{
		return t.key % 2 == 0;
	}
};

//	== sort ==

int sort(const uint8_t *data, size_t size)
{
	std::vector<uint8_t> working(data, data + size);
	std::sort(working.begin(), working.end());

	if (!std::is_sorted(working.begin(), working.end())) return 1;
	if (!std::is_permutation(data, data + size, working.begin())) return 99;
	return 0;
}


//	== stable_sort ==

int stable_sort(const uint8_t *data, size_t size)
{
	std::vector<stable_test> input;
	for (size_t i = 0; i < size; ++i)
		input.push_back(stable_test(data[i], i));
	std::vector<stable_test> working = input;
	std::stable_sort(working.begin(), working.end(), key_less());

	if (!std::is_sorted(working.begin(), working.end(), key_less()))   return 1;
	auto iter = working.begin();
	while (iter != working.end())
	{
		auto range = std::equal_range(iter, working.end(), *iter, key_less());
		if (!std::is_sorted(range.first, range.second, total_less())) return 2;			
		iter = range.second;
	}
	if (!std::is_permutation(input.begin(), input.end(), working.begin())) return 99;
	return 0;
}

//	== partition ==

int partition(const uint8_t *data, size_t size)
{
	std::vector<uint8_t> working(data, data + size);
	auto iter = std::partition(working.begin(), working.end(), is_even<uint8_t>());

	if (!std::all_of (working.begin(), iter, is_even<uint8_t>())) return 1;
	if (!std::none_of(iter,   working.end(), is_even<uint8_t>())) return 2;
	if (!std::is_permutation(data, data + size, working.begin())) return 99;
	return 0;
}


//	== stable_partition ==

int stable_partition (const uint8_t *data, size_t size)
{
	std::vector<stable_test> input;
	for (size_t i = 0; i < size; ++i)
		input.push_back(stable_test(data[i], i));
	std::vector<stable_test> working = input;
	auto iter = std::stable_partition(working.begin(), working.end(), is_even<stable_test>());

	if (!std::all_of (working.begin(), iter, is_even<stable_test>())) return 1;
	if (!std::none_of(iter,   working.end(), is_even<stable_test>())) return 2;
	if (!std::is_sorted(working.begin(), iter, payload_less()))   return 3;
	if (!std::is_sorted(iter,   working.end(), payload_less()))   return 4;
	if (!std::is_permutation(input.begin(), input.end(), working.begin())) return 99;
	return 0;
}

//	== nth_element ==
//	use the first element as a position into the data
int nth_element (const uint8_t *data, size_t size)
{
	if (size <= 1) return 0;
	const size_t partition_point = data[0] % size;	
	std::vector<uint8_t> working(data + 1, data + size);
	const auto partition_iter = working.begin() + partition_point;
	std::nth_element(working.begin(), partition_iter, working.end());

//	nth may be the end iterator, in this case nth_element has no effect.
	if (partition_iter == working.end())
	{
		if (!std::equal(data + 1, data + size, working.begin())) return 98;
	}
	else
	{
		const uint8_t nth = *partition_iter;
		if (!std::all_of(working.begin(), partition_iter, [=](uint8_t v) { return v <= nth; }))
			return 1;
		if (!std::all_of(partition_iter, working.end(),   [=](uint8_t v) { return v >= nth; }))
			return 2;
		if (!std::is_permutation(data + 1, data + size, working.begin())) return 99;
		}

	return 0;
}

//	== partial_sort ==
//	use the first element as a position into the data
int partial_sort (const uint8_t *data, size_t size)
{
	if (size <= 1) return 0;
	const size_t sort_point = data[0] % size;
	std::vector<uint8_t> working(data + 1, data + size);
	const auto sort_iter = working.begin() + sort_point;
	std::partial_sort(working.begin(), sort_iter, working.end());

	if (sort_iter != working.end())
	{
		const uint8_t nth = *std::min_element(sort_iter, working.end());
		if (!std::all_of(working.begin(), sort_iter, [=](uint8_t v) { return v <= nth; }))
			return 1;
		if (!std::all_of(sort_iter, working.end(),   [=](uint8_t v) { return v >= nth; }))
			return 2;		
	}
	if (!std::is_sorted(working.begin(), sort_iter)) return 3;
	if (!std::is_permutation(data + 1, data + size, working.begin())) return 99;

	return 0;
}


// --	regex fuzzers

static int regex_helper(const uint8_t *data, size_t size, std::regex::flag_type flag)
{
	if (size > 0)
	{
		try
		{
			std::string s((const char *)data, size);
			std::regex re(s, flag);
			return std::regex_match(s, re) ? 1 : 0;
		} 
		catch (std::regex_error &ex) {} 
	}
	return 0;		
}


int regex_ECMAScript (const uint8_t *data, size_t size)
{
	(void) regex_helper(data, size, std::regex_constants::ECMAScript);
	return 0;
}

int regex_POSIX (const uint8_t *data, size_t size)
{
	(void) regex_helper(data, size, std::regex_constants::basic);
	return 0;
}

int regex_extended (const uint8_t *data, size_t size)
{
	(void) regex_helper(data, size, std::regex_constants::extended);
	return 0;
}

int regex_awk (const uint8_t *data, size_t size)
{
	(void) regex_helper(data, size, std::regex_constants::awk);
	return 0;
}

int regex_grep (const uint8_t *data, size_t size)
{
	(void) regex_helper(data, size, std::regex_constants::grep);
	return 0;
}

int regex_egrep (const uint8_t *data, size_t size)
{
	(void) regex_helper(data, size, std::regex_constants::egrep);
	return 0;
}

} // namespace fuzzing
