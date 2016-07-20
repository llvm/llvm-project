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

#include "check.h"
#include "re.h"

namespace benchmark {

Regex::Regex() : init_(false) { }

bool Regex::Init(const std::string& spec, std::string* error) {
  int ec = regcomp(&re_, spec.c_str(), REG_EXTENDED | REG_NOSUB);
  if (ec != 0) {
    if (error) {
      size_t needed = regerror(ec, &re_, nullptr, 0);
      char* errbuf = new char[needed];
      regerror(ec, &re_, errbuf, needed);

      // regerror returns the number of bytes necessary to null terminate
      // the string, so we move that when assigning to error.
      CHECK_NE(needed, 0);
      error->assign(errbuf, needed - 1);

      delete[] errbuf;
    }

    return false;
  }

  init_ = true;
  return true;
}

Regex::~Regex() {
  if (init_) {
    regfree(&re_);
  }
}

bool Regex::Match(const std::string& str) {
  if (!init_) {
    return false;
  }

  return regexec(&re_, str.c_str(), 0, nullptr, 0) == 0;
}

}  // end namespace benchmark
