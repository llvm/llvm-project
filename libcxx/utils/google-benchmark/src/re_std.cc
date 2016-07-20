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

#include "re.h"

namespace benchmark {

Regex::Regex() : init_(false) { }

bool Regex::Init(const std::string& spec, std::string* error) {
  try {
    re_ = std::regex(spec, std::regex_constants::extended);

    init_ = true;
  } catch (const std::regex_error& e) {
    if (error) {
      *error = e.what();
    }
  }
  return init_;
}

Regex::~Regex() { }

bool Regex::Match(const std::string& str) {
  if (!init_) {
    return false;
  }

  return std::regex_search(str, re_);
}

}  // end namespace benchmark
