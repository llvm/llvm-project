#include <algorithm>
#include <array>
#include <cassert>
#include <ios>
#include <limits>
#include <span>
#include <spanstream>
#include <string_view>
#include <type_traits>
#include <utility>

#include <algorithm>
#include <cassert>
#include <span>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

#include <iostream>

using namespace std;

template <class Spanbuf>
class basic_test_buf : public Spanbuf {
public:
  using Spanbuf::Spanbuf;

  using Spanbuf::eback;
  using Spanbuf::egptr;
  using Spanbuf::epptr;
  using Spanbuf::gptr;
  using Spanbuf::pbase;
  using Spanbuf::pptr;

  using Spanbuf::setp;

  using Spanbuf::seekoff;
  using Spanbuf::seekpos;
  using Spanbuf::setbuf;
};

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using test_buf = basic_test_buf<basic_spanbuf<CharT>>;
  { // construction
    CharT buffer[10];
    const test_buf default_constructed{};
    assert(default_constructed.span().data() == nullptr);
    assert(default_constructed.eback() == nullptr);
    assert(default_constructed.gptr() == nullptr);
    assert(default_constructed.egptr() == nullptr);
    assert(default_constructed.pbase() == nullptr);
    assert(default_constructed.pptr() == nullptr);
    assert(default_constructed.epptr() == nullptr);

    const test_buf mode_constructed{ios_base::in};
    assert(mode_constructed.span().data() == nullptr);
    assert(mode_constructed.eback() == nullptr);
    assert(mode_constructed.gptr() == nullptr);
    assert(mode_constructed.egptr() == nullptr);
    assert(mode_constructed.pbase() == nullptr);
    assert(mode_constructed.pptr() == nullptr);
    assert(mode_constructed.epptr() == nullptr);

    test_buf span_constructed{span<CharT>{buffer}};
    assert(span_constructed.span().data() == buffer);
    assert(span_constructed.eback() == buffer);
    assert(span_constructed.gptr() == buffer);
    // assert(span_constructed.egptr() == std::end(buffer));
    assert(span_constructed.pbase() == buffer);
    assert(span_constructed.pptr() == buffer);
    // assert(span_constructed.epptr() == std::end(buffer));

    const test_buf span_mode_in_constructed{span<CharT>{buffer}, ios_base::in};
    assert(span_mode_in_constructed.span().data() == buffer);
    assert(span_mode_in_constructed.eback() == buffer);
    assert(span_mode_in_constructed.gptr() == buffer);
    // assert(span_mode_in_constructed.egptr() == std::end(buffer));
    assert(span_mode_in_constructed.pbase() == nullptr);
    assert(span_mode_in_constructed.pptr() == nullptr);
    assert(span_mode_in_constructed.epptr() == nullptr);

    const test_buf span_mode_in_ate_constructed{span<CharT>{buffer}, ios_base::in | ios_base::ate};
    assert(span_mode_in_ate_constructed.span().data() == buffer);
    assert(span_mode_in_ate_constructed.eback() == buffer);
    assert(span_mode_in_ate_constructed.gptr() == buffer);
    // assert(span_mode_in_ate_constructed.egptr() == std::end(buffer));
    assert(span_mode_in_ate_constructed.pbase() == nullptr);
    assert(span_mode_in_ate_constructed.pptr() == nullptr);
    assert(span_mode_in_ate_constructed.epptr() == nullptr);

    const test_buf span_mode_out_constructed{span<CharT>{buffer}, ios_base::out};
    assert(span_mode_out_constructed.span().data() == buffer);
    assert(span_mode_out_constructed.eback() == nullptr);
    assert(span_mode_out_constructed.gptr() == nullptr);
    assert(span_mode_out_constructed.egptr() == nullptr);
    assert(span_mode_out_constructed.pbase() == buffer);
    assert(span_mode_out_constructed.pptr() == buffer);
    assert(span_mode_out_constructed.epptr() == std::end(buffer));

    const test_buf span_mode_out_ate_constructed{span<CharT>{buffer}, ios_base::out | ios_base::ate};
    assert(span_mode_out_ate_constructed.span().data() == buffer);
    assert(span_mode_out_ate_constructed.eback() == nullptr);
    assert(span_mode_out_ate_constructed.gptr() == nullptr);
    assert(span_mode_out_ate_constructed.egptr() == nullptr);
    assert(span_mode_out_ate_constructed.pbase() == buffer);
    // assert(span_mode_out_ate_constructed.pptr() == std::end(buffer));
    // assert(span_mode_out_ate_constructed.epptr() == std::end(buffer));

    const test_buf span_mode_unknown_constructed{span<CharT>{buffer}, 0};
    assert(span_mode_unknown_constructed.span().data() == buffer);
    assert(span_mode_unknown_constructed.eback() == nullptr);
    assert(span_mode_unknown_constructed.gptr() == nullptr);
    assert(span_mode_unknown_constructed.egptr() == nullptr);
    assert(span_mode_unknown_constructed.pbase() == nullptr);
    assert(span_mode_unknown_constructed.pptr() == nullptr);
    assert(span_mode_unknown_constructed.epptr() == nullptr);
#if 0
    test_buf move_constructed{std::move(span_constructed)};
    assert(move_constructed.span().data() == buffer);
    assert(move_constructed.eback() == buffer);
    assert(move_constructed.gptr() == buffer);
    // assert(move_constructed.egptr() == std::end(buffer));
    assert(move_constructed.pbase() == buffer);
    assert(move_constructed.pptr() == buffer);
    // assert(move_constructed.epptr() == std::end(buffer));
    assert(span_constructed.span().data() == nullptr);
    assert(span_constructed.eback() == nullptr);
    assert(span_constructed.gptr() == nullptr);
    assert(span_constructed.egptr() == nullptr);
    assert(span_constructed.pbase() == nullptr);
    assert(span_constructed.pptr() == nullptr);
    assert(span_constructed.epptr() == nullptr);

    test_buf move_assigned;
    move_assigned = std::move(move_constructed);
    assert(move_assigned.span().data() == buffer);
    assert(move_assigned.eback() == buffer);
    assert(move_assigned.gptr() == buffer);
    // assert(move_assigned.egptr() == std::end(buffer));
    assert(move_assigned.pbase() == buffer);
    assert(move_assigned.pptr() == buffer);
    // assert(move_assigned.epptr() == std::end(buffer));
    assert(move_constructed.span().data() == nullptr);
    assert(move_constructed.eback() == nullptr);
    assert(move_constructed.gptr() == nullptr);
    assert(move_constructed.egptr() == nullptr);
    assert(move_constructed.pbase() == nullptr);
    assert(move_constructed.pptr() == nullptr);
    assert(move_constructed.epptr() == nullptr);
#endif
  }

  // { // swap
  //   CharT buffer1[10];
  //   CharT buffer2[20];
  //   test_buf first{span<CharT>{buffer1}};
  //   test_buf second{span<CharT>{buffer2}};
  //   assert(first.span().data() == buffer1);
  //   assert(second.span().data() == buffer2);

  //   first.swap(second);
  //   assert(first.span().data() == buffer2);
  //   assert(second.span().data() == buffer1);

  //   swap(first, second);
  //   assert(first.span().data() == buffer1);
  //   assert(second.span().data() == buffer2);
  // }

  // { // span, span, span, span
  //   CharT buffer1[10];
  //   test_buf input_buffer{span<CharT>{buffer1}, ios_base::in};
  //   assert(input_buffer.span().data() == buffer1);
  //   assert(input_buffer.span().size() == std::size(buffer1));

  //   test_buf output_buffer{span<CharT>{buffer1}, ios_base::out};
  //   assert(output_buffer.span().data() == buffer1);
  //   assert(output_buffer.span().size() == 0); // counts the written characters

  //   // Manually move the written pointer
  //   output_buffer.setp(buffer1, buffer1 + 5, std::end(buffer1));
  //   assert(output_buffer.span().data() == buffer1);
  //   assert(output_buffer.span().size() == 5);

  //   CharT buffer2[10];
  //   input_buffer.span(span<CharT>{buffer2});
  //   assert(input_buffer.span().data() == buffer2);
  //   assert(input_buffer.span().size() == std::size(buffer2));

  //   output_buffer.span(span<CharT>{buffer2});
  //   assert(output_buffer.span().data() == buffer2);
  //   assert(output_buffer.span().size() == 0);

  //   test_buf hungry_buffer{span<CharT>{buffer1}, ios_base::out | ios_base::ate};
  //   assert(hungry_buffer.span().data() == buffer1);
  //   assert(hungry_buffer.span().size() == std::size(buffer1));

  //   hungry_buffer.span(span<CharT>{buffer2});
  //   assert(hungry_buffer.span().data() == buffer2);
  //   assert(hungry_buffer.span().size() == std::size(buffer2));
  // }

  { // seekoff ios_base::beg
    CharT buffer[10];
    test_buf input_buffer{span<CharT>{buffer}, ios_base::in};
    test_buf output_buffer{span<CharT>{buffer}, ios_base::out};

    auto result = input_buffer.seekoff(0, ios_base::beg, ios_base::in);
    assert(result == 0);

    // pptr not set but off is 0
    result = input_buffer.seekoff(0, ios_base::beg, ios_base::out);
    assert(result == 0);

    // pptr not set and off != 0 -> fail
    result = input_buffer.seekoff(1, ios_base::beg, ios_base::out);
    assert(result == -1);

    // gptr not set but off is 0
    result = output_buffer.seekoff(0, ios_base::beg, ios_base::in);
    assert(result == 0);

    // gptr not set and off != 0 -> fail
    result = output_buffer.seekoff(1, ios_base::beg, ios_base::in);
    assert(result == -1);

    // negative off -> fail
    result = input_buffer.seekoff(-1, ios_base::beg, ios_base::in);
    assert(result == -1);

    // negative off -> fail
    result = output_buffer.seekoff(-1, ios_base::beg, ios_base::out);
    assert(result == -1);

    // off larger than buf -> fail
    result = input_buffer.seekoff(20, ios_base::beg, ios_base::in);
    assert(result == -1);

    // off larger than buf -> fail
    result = output_buffer.seekoff(20, ios_base::beg, ios_base::out);
    assert(result == -1);

    // passes
    result = input_buffer.seekoff(5, ios_base::beg, ios_base::in);
    assert(result == 5);

    result = output_buffer.seekoff(5, ios_base::beg, ios_base::out);
    assert(result == 5);

    // always from front
    result = input_buffer.seekoff(7, ios_base::beg, ios_base::in);
    assert(result == 7);

    result = output_buffer.seekoff(7, ios_base::beg, ios_base::out);
    assert(result == 7);
  }

  { // seekoff ios_base::end
    CharT buffer[10];
    test_buf input_buffer{span<CharT>{buffer}, ios_base::in};
    // all fine we move to end of stream
    auto result = input_buffer.seekoff(0, ios_base::end, ios_base::in);
    assert(result == 10);

    // pptr not set but off is == 0
    result = input_buffer.seekoff(-10, ios_base::end, ios_base::out);
    std::cerr << result << std::endl;
    assert(result == 0);

    // pptr not set and off != 0 -> fail
    result = input_buffer.seekoff(0, ios_base::end, ios_base::out);
    std::cerr << result << std::endl;
    assert(result == -1);

    // negative off -> fail
    result = input_buffer.seekoff(-20, ios_base::end, ios_base::in);
    assert(result == -1);

    // off beyond end of buffer -> fail
    result = input_buffer.seekoff(1, ios_base::end, ios_base::in);
    assert(result == -1);

    // passes and moves to buffer size - off
    result = input_buffer.seekoff(-5, ios_base::end, ios_base::in);
    assert(result == 5);

    // always from front
    result = input_buffer.seekoff(-7, ios_base::end, ios_base::in);
    assert(result == 3);

    // integer overflow due to large off
    result = input_buffer.seekoff(numeric_limits<long long>::max(), ios_base::end, ios_base::in);
    assert(result == -1);

    test_buf output_buffer{span<CharT>{buffer}, ios_base::out};
    // gptr not set but off is 0
    result = output_buffer.seekoff(0, ios_base::end, ios_base::in);
    assert(result == 0);

    // newoff is negative -> fail
    result = output_buffer.seekoff(-10, ios_base::end, ios_base::out);
    assert(result == -1);

    // pptr not set but off == 0
    result = output_buffer.seekoff(0, ios_base::end, ios_base::out);
    assert(result == 0);

    // all fine we stay at end of stream
    result = output_buffer.seekoff(0, ios_base::end, ios_base::in);
    assert(result == 0);

    // gptr not set and off != 0 -> fail
    result = output_buffer.seekoff(1, ios_base::end, ios_base::in);
    assert(result == -1);

    // off + buffer size is negative -> fail
    result = output_buffer.seekoff(-20, ios_base::end, ios_base::out);
    assert(result == -1);

    // off larger than buffer -> fail
    result = output_buffer.seekoff(11, ios_base::end, ios_base::out);
    assert(result == -1);

    // passes and moves to buffer size - off
    result = output_buffer.seekoff(5, ios_base::end, ios_base::out);
    assert(result == 5);

    // passes we are still below buffer size
    result = output_buffer.seekoff(3, ios_base::end, ios_base::out);
    assert(result == 8);

    // moves beyond buffer size -> fails
    result = output_buffer.seekoff(3, ios_base::end, ios_base::out);
    assert(result == -1);

    // integer overflow due to large off
    result = output_buffer.seekoff(numeric_limits<long long>::max(), ios_base::end, ios_base::in);
    assert(result == -1);

    test_buf inout_buffer{span<CharT>{buffer}, ios_base::in | ios_base::out};
    // all fine we move to end of stream
    result = inout_buffer.seekoff(0, ios_base::end, ios_base::in);
    assert(result == 10);

    // we move to front of the buffer
    result = inout_buffer.seekoff(-10, ios_base::end, ios_base::out);
    assert(result == 0);

    // we move to end of buffer
    result = inout_buffer.seekoff(0, ios_base::end, ios_base::out);
    assert(result == 10);

    // negative off -> fail
    result = inout_buffer.seekoff(-20, ios_base::end, ios_base::in);
    assert(result == -1);

    // off beyond end of buffer -> fail
    result = inout_buffer.seekoff(1, ios_base::end, ios_base::in);
    assert(result == -1);

    // passes and moves to buffer size - off
    result = inout_buffer.seekoff(-5, ios_base::end, ios_base::in);
    assert(result == 5);

    // always from front
    result = inout_buffer.seekoff(-7, ios_base::end, ios_base::in);
    assert(result == 3);

    // integer overflow due to large off
    result = inout_buffer.seekoff(numeric_limits<long long>::max(), ios_base::end, ios_base::in);
    assert(result == -1);
  }

  { // seekoff ios_base::cur
    CharT buffer[10];
    test_buf input_buffer{span<CharT>{buffer}, ios_base::in};

    // no mode set -> fail
    auto result = input_buffer.seekoff(0, ios_base::cur, 0);
    std::cerr << result << std::endl;
    assert(result == -1);

    // both in and out modes set -> fail
    result = input_buffer.seekoff(0, ios_base::cur, ios_base::in | ios_base::out);
    assert(result == -1);

    // pptr not set and off is != 0 -> fail
    result = input_buffer.seekoff(1, ios_base::cur, ios_base::out);
    assert(result == -1);

    // off larger than buffer size -> fail
    result = input_buffer.seekoff(20, ios_base::cur, ios_base::out);
    assert(result == -1);

    // off negative -> fail
    result = input_buffer.seekoff(-1, ios_base::cur, ios_base::out);
    assert(result == -1);

    // pptr not set but off is == 0
    result = input_buffer.seekoff(0, ios_base::cur, ios_base::out);
    assert(result == 0);

    // passes and sets position
    result = input_buffer.seekoff(3, ios_base::cur, ios_base::in);
    assert(result == 3);

    // negative off moves back
    result = input_buffer.seekoff(-2, ios_base::cur, ios_base::in);
    assert(result == 1);

    // off + current position is beyond buffer size -> fail
    result = input_buffer.seekoff(10, ios_base::cur, ios_base::in);
    assert(result == -1);

    test_buf output_buffer{span<CharT>{buffer}, ios_base::out};
    // no mode set -> fail
    result = output_buffer.seekoff(0, ios_base::cur, 0);
    std::cerr << result << std::endl;
    // assert(result == -1);

    // both in and out modes set -> fail
    result = output_buffer.seekoff(0, ios_base::cur, ios_base::in | ios_base::out);
    assert(result == -1);

    // gptr not set and off is != 0 -> fail
    result = output_buffer.seekoff(1, ios_base::cur, ios_base::in);
    assert(result == -1);

    // off larger than buffer size -> fail
    result = output_buffer.seekoff(20, ios_base::cur, ios_base::out);
    assert(result == -1);

    // off negative -> fail
    result = output_buffer.seekoff(-1, ios_base::cur, ios_base::out);
    assert(result == -1);

    // gptr not set but off is == 0
    result = output_buffer.seekoff(0, ios_base::cur, ios_base::in);
    assert(result == 0);

    // passes and sets position
    result = output_buffer.seekoff(3, ios_base::cur, ios_base::out);
    assert(result == 3);

    // negative off moves back
    result = output_buffer.seekoff(-2, ios_base::cur, ios_base::out);
    assert(result == 1);

    // off + current position is beyond buffer size -> fail
    result = output_buffer.seekoff(10, ios_base::cur, ios_base::out);
    assert(result == -1);

    test_buf inout_buffer{span<CharT>{buffer}, ios_base::in | ios_base::out};
    // no mode set -> fail
    result = inout_buffer.seekoff(0, ios_base::cur, 0);
    std::cerr << result << std::endl;
    // assert(result == -1);

    // both in and out modes set -> fail
    result = inout_buffer.seekoff(0, ios_base::cur, ios_base::in | ios_base::out);
    assert(result == -1);

    // off larger than buffer size -> fail
    result = inout_buffer.seekoff(20, ios_base::cur, ios_base::out);
    assert(result == -1);

    // off negative -> fail
    result = inout_buffer.seekoff(-1, ios_base::cur, ios_base::out);
    assert(result == -1);

    // Moves input sequence to position 3
    result = inout_buffer.seekoff(3, ios_base::cur, ios_base::in);
    assert(result == 3);

    // Moves output sequence to position 3
    result = inout_buffer.seekoff(3, ios_base::cur, ios_base::out);
    assert(result == 3);

    // negative off moves back
    result = inout_buffer.seekoff(-2, ios_base::cur, ios_base::in);
    assert(result == 1);

    // negative off moves back
    result = inout_buffer.seekoff(-2, ios_base::cur, ios_base::out);
    assert(result == 1);

    // off + current position is beyond buffer size -> fail
    result = inout_buffer.seekoff(10, ios_base::cur, ios_base::in);
    assert(result == -1);

    // off + current position is beyond buffer size -> fail
    result = inout_buffer.seekoff(10, ios_base::cur, ios_base::out);
    assert(result == -1);

    // off + current position is before buffer size -> fail
    result = inout_buffer.seekoff(-2, ios_base::cur, ios_base::in);
    assert(result == -1);

    // off + current position is before buffer size -> fail
    result = inout_buffer.seekoff(-2, ios_base::cur, ios_base::out);
    assert(result == -1);
  }

  { // seekpos (same as seekoff with ios_base::beg)
    CharT buffer[10];
    test_buf input_buffer{span<CharT>{buffer}, ios_base::in};
    test_buf output_buffer{span<CharT>{buffer}, ios_base::out};

    auto result = input_buffer.seekpos(0, ios_base::in);
    assert(result == 0);

    // pptr not set but off is 0
    result = input_buffer.seekpos(0, ios_base::out);
    assert(result == 0);

    // pptr not set and off != 0 -> fail
    result = input_buffer.seekpos(1, ios_base::out);
    assert(result == -1);

    // gptr not set but off is 0
    result = output_buffer.seekpos(0, ios_base::in);
    assert(result == 0);

    // gptr not set and off != 0 -> fail
    result = output_buffer.seekpos(1, ios_base::in);
    assert(result == -1);

    // negative off -> fail
    result = input_buffer.seekpos(-1, ios_base::in);
    assert(result == -1);

    // negative off -> fail
    result = output_buffer.seekpos(-1, ios_base::out);
    assert(result == -1);

    // off larger than buf -> fail
    result = input_buffer.seekpos(20, ios_base::in);
    assert(result == -1);

    // off larger than buf -> fail
    result = output_buffer.seekpos(20, ios_base::out);
    assert(result == -1);

    // passes
    result = input_buffer.seekpos(5, ios_base::in);
    assert(result == 5);

    result = output_buffer.seekpos(5, ios_base::out);
    assert(result == 5);

    // always from front
    result = input_buffer.seekpos(7, ios_base::in);
    assert(result == 7);

    result = output_buffer.seekpos(7, ios_base::out);
    assert(result == 7);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  // test<nasty_char, nasty_char_traits>();
#endif
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
