//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

#include <unordered_map>
#include <string>
#include <string_view>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <iostream>
#include <vector>
#include <random>

struct StringHash {
    std::size_t operator()(const std::string& str) const {
        return string_hash(str);
    }

    std::size_t operator()(const std::string_view& str_view) const {
        return string_view_hash(str_view);
    }

    std::hash<std::string> string_hash;
    std::hash<std::string_view> string_view_hash;
};

struct StringHashTransparent : StringHash {
    using is_transparent = void;
};

struct StringEq {
    template <typename Str1, typename Str2>
    bool operator()(const Str1& str1, const Str2& str2) const {
        return std::equal(str1.begin(), str1.end(), str2.begin(), str2.end());
    }
};

struct StringEqTransparent : StringEq {
    using is_transparent = void;
};

using umap_type = std::unordered_map<std::string, int, StringHash, StringEq>;
using umap_type_hetero = std::unordered_map<std::string, int, StringHashTransparent, StringEqTransparent>;
std::size_t num_measurements = 1001;
std::size_t num_insertions = 100000;
std::size_t string_size = 10;

template <typename MapType>
double test_try_emplace() {
    std::string insertion_string(string_size, 'a');
    std::string_view insertion_view = insertion_string;
    std::vector<double> measurements;
    measurements.reserve(num_measurements);

    for (std::size_t i = 0; i < num_measurements; ++i) {
        MapType umap;

        auto start = std::chrono::steady_clock::now();

        for (std::size_t j = 0; j < num_insertions; ++j) {
            if constexpr (std::is_same_v<MapType, umap_type>) {
                umap.try_emplace(std::string(insertion_view), 1);
            } else {
                umap.try_emplace(insertion_view, 1);
            }
        }

        auto finish = std::chrono::steady_clock::now();

        assert(umap.size() == 1);
        std::chrono::duration<double> duration = finish - start;
        // std::cerr << "\tElapsed time #" << i << " is " << duration.count() << std::endl;
        measurements.emplace_back(duration.count());
    }

    std::sort(measurements.begin(), measurements.end());
    std::cerr << std::endl;
    std::cerr << "\tLowest time " << measurements.front() << std::endl;
    std::cerr << "\tMedian time " << measurements[num_measurements / 2] << std::endl;
    std::cerr << "\tHighest time " << measurements.back() << std::endl << std::endl;

    return measurements[num_measurements / 2];
}

template <typename MapType>
double test_try_emplace_random(const std::vector<std::string>& random_strings) {
    std::vector<double> measurements;
    measurements.reserve(num_measurements);

    for (std::size_t i = 0; i < num_measurements; ++i) {
        MapType umap(random_strings.size());

        auto start = std::chrono::steady_clock::now();

        for (auto& str : random_strings) {
            std::string_view random_string_view = str;
            if constexpr (std::is_same_v<MapType, umap_type>) {
                umap.try_emplace(std::string(random_string_view), 1);
            } else {
                umap.try_emplace(random_string_view, 1);
            }
        }

        auto finish = std::chrono::steady_clock::now();

        std::chrono::duration<double> duration = finish - start;
        // std::cerr << "\tElapsed time #" << i << " is " << duration.count() << std::endl;
        measurements.emplace_back(duration.count());
    }
    std::sort(measurements.begin(), measurements.end());
    std::cerr << std::endl << "\tMedian time " << measurements[num_measurements / 2] << std::endl << std::endl;
    return measurements[num_measurements / 2];
}

int main() {
    std::cerr << "Test homogeneous overloads single" << std::endl;
    double homogeneous_result = test_try_emplace<umap_type>();
    std::cerr << "test heterogeneous overloads single" << std::endl;
    double heterogeneous_result = test_try_emplace<umap_type_hetero>();

    std::cerr << "homo/hetero single = " << homogeneous_result / heterogeneous_result << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 100);

    std::vector<std::string> random_vector;
    random_vector.reserve(num_insertions);

    for (std::size_t i = 0; i < num_insertions; ++i) {
        random_vector.emplace_back(string_size, char(distrib(gen)));
    }

    std::cerr << std::endl << "Test homogeneous overloads random" << std::endl;
    double homogeneous_result_random = test_try_emplace_random<umap_type>(random_vector);
    std::cerr << "Test heterogeneous overloads random"<< std::endl;
    double heterogeneous_result_random = test_try_emplace_random<umap_type_hetero>(random_vector);

    std::cerr << "homo/hetero random = " << homogeneous_result_random / heterogeneous_result_random << std::endl;

    return 1;
}