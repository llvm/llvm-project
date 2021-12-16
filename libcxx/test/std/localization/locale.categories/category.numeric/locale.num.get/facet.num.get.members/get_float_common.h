#ifndef GET_FLOAT_COMMON_H
#define GET_FLOAT_COMMON_H

/// Read a double from the input string, check that the expected number of
/// characters are read, the expected value is returned, and the expected
/// error is set.
#define TEST(STR, EXPECTED_LEN, EXPECTED_VAL, EXPECTED_ERR)                                                            \
  {                                                                                                                    \
    std::ios_base::iostate err = ios.goodbit;                                                                          \
    cpp17_input_iterator<const char*> iter =                                                                           \
        f.get(cpp17_input_iterator<const char*>((STR)), cpp17_input_iterator<const char*>((STR) + strlen((STR))), ios, \
              err, v);                                                                                                 \
    assert(iter.base() == (STR) + (EXPECTED_LEN) && "read wrong number of characters");                                \
    assert(err == (EXPECTED_ERR));                                                                                     \
    if (std::isnan(EXPECTED_VAL))                                                                                      \
      assert(std::isnan(v) && "expected NaN value");                                                                   \
    else                                                                                                               \
      assert(v == (EXPECTED_VAL) && "wrong value");                                                                    \
  }

#endif
