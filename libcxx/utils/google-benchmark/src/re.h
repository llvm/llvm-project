// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BENCHMARK_RE_H_
#define BENCHMARK_RE_H_

#if defined(HAVE_STD_REGEX)
#include <regex>
#elif defined(HAVE_GNU_POSIX_REGEX)
#include <gnuregex.h>
#elif defined(HAVE_POSIX_REGEX)
#include <regex.h>
#else
#error No regular expression backend was found!
#endif
#include <string>

namespace benchmark {

// A wrapper around the POSIX regular expression API that provides automatic
// cleanup
class Regex {
 public:
  Regex();
  ~Regex();

  // Compile a regular expression matcher from spec.  Returns true on success.
  //
  // On failure (and if error is not nullptr), error is populated with a human
  // readable error message if an error occurs.
  bool Init(const std::string& spec, std::string* error);

  // Returns whether str matches the compiled regular expression.
  bool Match(const std::string& str);
 private:
  bool init_;
  // Underlying regular expression object
#if defined(HAVE_STD_REGEX)
  std::regex re_;
#elif defined(HAVE_POSIX_REGEX) || defined(HAVE_GNU_POSIX_REGEX)
  regex_t re_;
#else
# error No regular expression backend implementation available
#endif
};

}  // end namespace benchmark

#endif  // BENCHMARK_RE_H_
