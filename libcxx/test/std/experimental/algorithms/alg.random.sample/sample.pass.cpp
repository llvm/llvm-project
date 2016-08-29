//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class PopulationIterator, class SampleIterator, class Distance,
//           class UniformRandomNumberGenerator>
// SampleIterator sample(PopulationIterator first, PopulationIterator last,
//                       SampleIterator out, Distance n,
//                       UniformRandomNumberGenerator &&g);

#include <experimental/algorithm>
#include <random>
#include <cassert>

#include "test_iterators.h"

struct ReservoirSampleExpectations {
  enum { os = 4 };
  static int oa1[os];
  static int oa2[os];
};

int ReservoirSampleExpectations::oa1[] = {10, 5, 9, 4};
int ReservoirSampleExpectations::oa2[] = {5, 2, 10, 4};

struct SelectionSampleExpectations {
  enum { os = 4 };
  static int oa1[os];
  static int oa2[os];
};

int SelectionSampleExpectations::oa1[] = {1, 4, 6, 7};
int SelectionSampleExpectations::oa2[] = {1, 2, 6, 8};

template <class IteratorCategory> struct TestExpectations
    : public SelectionSampleExpectations {};

template <>
struct TestExpectations<std::input_iterator_tag>
    : public ReservoirSampleExpectations {};

template <template<class...> class PopulationIteratorType, class PopulationItem,
          template<class...> class SampleIteratorType, class SampleItem>
void test() {
  typedef PopulationIteratorType<PopulationItem *> PopulationIterator;
  typedef SampleIteratorType<SampleItem *> SampleIterator;
  PopulationItem ia[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const unsigned is = sizeof(ia) / sizeof(ia[0]);
  typedef TestExpectations<typename std::iterator_traits<
      PopulationIterator>::iterator_category> Expectations;
  const unsigned os = Expectations::os;
  SampleItem oa[os];
  const int *oa1 = Expectations::oa1;
  const int *oa2 = Expectations::oa2;
  std::minstd_rand g;
  SampleIterator end;
  end = std::experimental::sample(PopulationIterator(ia),
                                  PopulationIterator(ia + is),
                                  SampleIterator(oa), os, g);
  assert(end.base() - oa == std::min(os, is));
  assert(std::equal(oa, oa + os, oa1));
  end = std::experimental::sample(PopulationIterator(ia),
                                  PopulationIterator(ia + is),
                                  SampleIterator(oa), os, std::move(g));
  assert(end.base() - oa == std::min(os, is));
  assert(std::equal(oa, oa + os, oa2));
}

template <template<class...> class PopulationIteratorType, class PopulationItem,
          template<class...> class SampleIteratorType, class SampleItem>
void test_empty_population() {
  typedef PopulationIteratorType<PopulationItem *> PopulationIterator;
  typedef SampleIteratorType<SampleItem *> SampleIterator;
  PopulationItem ia[] = {42};
  const unsigned os = 4;
  SampleItem oa[os];
  std::minstd_rand g;
  SampleIterator end =
      std::experimental::sample(PopulationIterator(ia), PopulationIterator(ia),
                                SampleIterator(oa), os, g);
  assert(end.base() == oa);
}

template <template<class...> class PopulationIteratorType, class PopulationItem,
          template<class...> class SampleIteratorType, class SampleItem>
void test_empty_sample() {
  typedef PopulationIteratorType<PopulationItem *> PopulationIterator;
  typedef SampleIteratorType<SampleItem *> SampleIterator;
  PopulationItem ia[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const unsigned is = sizeof(ia) / sizeof(ia[0]);
  SampleItem oa[1];
  std::minstd_rand g;
  SampleIterator end =
      std::experimental::sample(PopulationIterator(ia), PopulationIterator(ia + is),
                                SampleIterator(oa), 0, g);
  assert(end.base() == oa);
}

template <template<class...> class PopulationIteratorType, class PopulationItem,
          template<class...> class SampleIteratorType, class SampleItem>
void test_small_population() {
  // The population size is less than the sample size.
  typedef PopulationIteratorType<PopulationItem *> PopulationIterator;
  typedef SampleIteratorType<SampleItem *> SampleIterator;
  PopulationItem ia[] = {1, 2, 3, 4, 5};
  const unsigned is = sizeof(ia) / sizeof(ia[0]);
  const unsigned os = 8;
  SampleItem oa[os];
  const SampleItem oa1[] = {1, 2, 3, 4, 5};
  std::minstd_rand g;
  SampleIterator end;
  end = std::experimental::sample(PopulationIterator(ia),
                                  PopulationIterator(ia + is),
                                  SampleIterator(oa), os, g);
  assert(end.base() - oa == std::min(os, is));
  assert(std::equal(oa, end.base(), oa1));
}

int main() {
  test<input_iterator, int, random_access_iterator, int>();
  test<forward_iterator, int, output_iterator, int>();
  test<forward_iterator, int, random_access_iterator, int>();

  test<input_iterator, int, random_access_iterator, double>();
  test<forward_iterator, int, output_iterator, double>();
  test<forward_iterator, int, random_access_iterator, double>();

  test_empty_population<input_iterator, int, random_access_iterator, int>();
  test_empty_population<forward_iterator, int, output_iterator, int>();
  test_empty_population<forward_iterator, int, random_access_iterator, int>();

  test_empty_sample<input_iterator, int, random_access_iterator, int>();
  test_empty_sample<forward_iterator, int, output_iterator, int>();
  test_empty_sample<forward_iterator, int, random_access_iterator, int>();

  test_small_population<input_iterator, int, random_access_iterator, int>();
  test_small_population<forward_iterator, int, output_iterator, int>();
  test_small_population<forward_iterator, int, random_access_iterator, int>();
}
