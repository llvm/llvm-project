//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <ranges>
#include <string>

using namespace std::string_view_literals;

enum class Result {
  success,
  mismatch,
  no_match_found,
  unknown_matcher,
  invalid_use,
};

namespace co {
namespace {
[[noreturn]] void exit(Result result) { std::exit(static_cast<int>(result)); }

[[noreturn]] void print_failure(int line, std::string_view stdin_content, std::string_view matcher) {
  std::cout << "Failed to match: `" << matcher << "`\nRemaining data:\n" << stdin_content << '\n';
  co::exit(Result::mismatch);
}

bool is_newline(char c) { return c == '\n'; }

bool isblank(char c) { return std::isblank(c); }

bool consume_front(std::string_view& sv, std::string_view start) {
  if (!sv.starts_with(start))
    return false;
  sv.remove_prefix(start.size());
  return true;
}
} // namespace
} // namespace co

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "check_output has to be used as `<command> | ./check_output %s`\n";
    co::exit(Result::invalid_use);
  }

  std::string file_content_data = [&] {
    std::ifstream file(argv[1]);
    if (!file) {
      std::cerr << "Failed to open file: " << argv[1] << '\n';
      co::exit(Result::invalid_use);
    }
    return std::string{std::istreambuf_iterator<char>{file}, {}};
  }();
  std::string_view file_content = file_content_data; // Don't copy the data around all the time

  std::string stdin_content_data = [&] {
    std::cin >> std::noskipws;
    return std::string{std::istream_iterator<char>{std::cin}, {}};
  }();
  std::string_view stdin_content = stdin_content_data; // Don't copy the data around all the time

  size_t match_count = 0;
  auto drop_blanks   = std::views::drop_while(co::isblank);

  while (!file_content.empty()) {
    auto marker = std::ranges::search(file_content, "// CHECK"sv);
    if (marker.empty()) {
      if (match_count == 0) {
        std::cerr << "No matcher found!\n";
        co::exit(Result::no_match_found);
      }
      co::exit(Result::success);
    }
    file_content.remove_prefix(marker.end() - file_content.begin());

    const auto get_match = [&]() {
      return std::string_view(file_content.begin(), std::ranges::find(file_content, '\n'));
    };

    if (co::consume_front(file_content, ":")) {
      auto match = get_match();
      auto found = std::ranges::search(
          stdin_content | std::views::drop_while(std::not_fn(co::is_newline)) | std::views::drop(1),
          match | drop_blanks);
      if (found.empty()) {
        co::print_failure(1, stdin_content, match);
      }
      ++match_count;
      stdin_content.remove_prefix(found.end() - stdin_content.begin());
    } else if (co::consume_front(file_content, "-SAME:")) {
      auto match    = get_match();
      auto haystack = std::string_view(stdin_content.begin(), std::ranges::find(stdin_content, '\n'));
      auto found    = std::ranges::search(haystack, match | drop_blanks);
      if (found.empty()) {
        co::print_failure(1, stdin_content, match);
      }
      stdin_content.remove_prefix(found.end() - stdin_content.begin());
    } else if (co::consume_front(file_content, "-NEXT:")) {
      auto match    = get_match();
      auto haystack = [&] {
        auto begin = std::ranges::find(stdin_content, '\n');
        if (begin == stdin_content.end()) {
          co::print_failure(1, stdin_content, match);
        }
        ++begin;
        return std::string_view(begin, std::ranges::find(begin, stdin_content.end(), '\n'));
      }();
      auto found = std::ranges::search(haystack, match | drop_blanks);
      if (found.empty())
        co::print_failure(1, stdin_content, match);
      stdin_content.remove_prefix(found.end() - stdin_content.begin());
    } else {
      std::cerr << "Unkown matcher type "
                << std::string_view(file_content.begin(), std::ranges::find(file_content, ':')) << " found\n";
      co::exit(Result::unknown_matcher);
    }
  }

  co::exit(Result::success);
}
