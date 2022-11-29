//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_SEQUENCES_VECTOR_BOOL_VECTOR_BOOL_FMT_FORMAT_FUNCTIONS_TESTS_H
#define TEST_STD_CONTAINERS_SEQUENCES_VECTOR_BOOL_VECTOR_BOOL_FMT_FORMAT_FUNCTIONS_TESTS_H

#include <vector>

#include "format.functions.common.h"
#include "test_macros.h"

template <class CharT, class TestFunction, class ExceptionTest>
void format_tests(TestFunction check, ExceptionTest check_exception) {
  std::vector input{true, true, false};

  check(SV("[true, true, false]"), SV("{}"), input);

  // ***** underlying has no format-spec

  // *** align-fill & width ***
  check(SV("[true, true, false]     "), SV("{:24}"), input);
  check(SV("[true, true, false]*****"), SV("{:*<24}"), input);
  check(SV("__[true, true, false]___"), SV("{:_^24}"), input);
  check(SV("#####[true, true, false]"), SV("{:#>24}"), input);

  check(SV("[true, true, false]     "), SV("{:{}}"), input, 24);
  check(SV("[true, true, false]*****"), SV("{:*<{}}"), input, 24);
  check(SV("__[true, true, false]___"), SV("{:_^{}}"), input, 24);
  check(SV("#####[true, true, false]"), SV("{:#>{}}"), input, 24);

  check_exception("The format-spec range-fill field contains an invalid character", SV("{:}<}"), input);
  check_exception("The format-spec range-fill field contains an invalid character", SV("{:{<}"), input);
  check_exception("The format-spec range-fill field contains an invalid character", SV("{::<}"), input);

  // *** sign ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:-}"), input);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:+}"), input);
  check_exception("The format-spec should consume the input or end with a '}'", SV("{: }"), input);

  // *** alternate form ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:#}"), input);

  // *** zero-padding ***
  check_exception("A format-spec width field shouldn't have a leading zero", SV("{:0}"), input);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:.}"), input);

  // *** locale-specific form ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{:L}"), input);

  // *** n
  check(SV("__true, true, false___"), SV("{:_^22n}"), input);

  // *** type ***
  check_exception("The range-format-spec type m requires two elements for a pair or tuple", SV("{:m}"), input);
  check_exception("The range-format-spec type s requires formatting a character type", SV("{:s}"), input);
  check_exception("The range-format-spec type ?s requires formatting a character type", SV("{:?s}"), input);

  for (std::basic_string_view<CharT> fmt : fmt_invalid_types<CharT>("s"))
    check_exception("The format-spec should consume the input or end with a '}'", fmt, input);

  // ***** Only underlying has a format-spec
  check(SV("[true   , true   , false  ]"), SV("{::7}"), input);
  check(SV("[true***, true***, false**]"), SV("{::*<7}"), input);
  check(SV("[_true__, _true__, _false_]"), SV("{::_^7}"), input);
  check(SV("[:::true, :::true, ::false]"), SV("{:::>7}"), input);

  check(SV("[true   , true   , false  ]"), SV("{::{}}"), input, 7);
  check(SV("[true***, true***, false**]"), SV("{::*<{}}"), input, 7);
  check(SV("[_true__, _true__, _false_]"), SV("{::_^{}}"), input, 7);
  check(SV("[:::true, :::true, ::false]"), SV("{:::>{}}"), input, 7);

  check_exception("The format-spec fill field contains an invalid character", SV("{::}<}"), input);
  check_exception("The format-spec fill field contains an invalid character", SV("{::{<}"), input);

  // *** sign ***
  check_exception("A sign field isn't allowed in this format-spec", SV("{::-}"), input);
  check_exception("A sign field isn't allowed in this format-spec", SV("{::+}"), input);
  check_exception("A sign field isn't allowed in this format-spec", SV("{:: }"), input);

  check(SV("[1, 1, 0]"), SV("{::-d}"), input);
  check(SV("[+1, +1, +0]"), SV("{::+d}"), input);
  check(SV("[ 1,  1,  0]"), SV("{:: d}"), input);

  // *** alternate form ***
  check_exception("An alternate form field isn't allowed in this format-spec", SV("{::#}"), input);

  check(SV("[0x1, 0x1, 0x0]"), SV("{::#x}"), input);

  // *** zero-padding ***
  check_exception("A zero-padding field isn't allowed in this format-spec", SV("{::05}"), input);

  check(SV("[00001, 00001, 00000]"), SV("{::05o}"), input);

  // *** precision ***
  check_exception("The format-spec should consume the input or end with a '}'", SV("{::.}"), input);

  // *** type ***
  for (std::basic_string_view<CharT> fmt : fmt_invalid_nested_types<CharT>("bBdosxX"))
    check_exception("The format-spec type has a type not supported for a bool argument", fmt, input);

  // ***** Both have a format-spec
  check(SV("^^[:::true, :::true, ::false]^^^"), SV("{:^^32::>7}"), input);
  check(SV("^^[:::true, :::true, ::false]^^^"), SV("{:^^{}::>7}"), input, 32);
  check(SV("^^[:::true, :::true, ::false]^^^"), SV("{:^^{}::>{}}"), input, 32, 7);

  check_exception("Argument index out of bounds", SV("{:^^{}::>5}"), input);
  check_exception("Argument index out of bounds", SV("{:^^{}::>{}}"), input, 32);
}

#endif // TEST_STD_CONTAINERS_SEQUENCES_VECTOR_BOOL_VECTOR_BOOL_FMT_FORMAT_FUNCTIONS_TESTS_H
