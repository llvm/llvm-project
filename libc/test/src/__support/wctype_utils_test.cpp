#include "src/__support/macros/config.h"
#include "src/__support/wctype/wctype_classification_utils.h"
#include "test/UnitTest/Test.h"

namespace {

namespace ascii_mode {
#undef LIBC_CONF_WCTYPE_MODE
#define LIBC_CONF_WCTYPE_MODE LIBC_WCTYPE_MODE_ASCII

#undef LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
#include "src/__support/wctype_utils.h"
} // namespace ascii_mode

namespace utf8_mode {
#undef LIBC_CONF_WCTYPE_MODE
#define LIBC_CONF_WCTYPE_MODE LIBC_WCTYPE_MODE_UTF8

namespace LIBC_NAMESPACE_DECL {
using ::LIBC_NAMESPACE::lookup_properties;
using ::LIBC_NAMESPACE::PropertyFlag;
} // namespace LIBC_NAMESPACE_DECL

#undef LLVM_LIBC_SRC___SUPPORT_WCTYPE_UTILS_H
#include "src/__support/wctype_utils.h"
} // namespace utf8_mode

struct TestCase {
  uint32_t wc;
  const char *name;
  bool expected;
};

// Helper function to mark the sections of the ASCII table that are
// punctuation characters. These are listed below:
//  Decimal    |         Symbol
//  -----------------------------------------
//  33 -  47   |  ! " $ % & ' ( ) * + , - . /
//  58 -  64   |  : ; < = > ? @
//  91 -  96   |  [ \ ] ^ _ `
// 123 - 126   |  { | } ~
bool is_punctuation_character(int c) {
  return ('!' <= c && c <= '/') || (':' <= c && c <= '@') ||
         ('[' <= c && c <= '`') || ('{' <= c && c <= '~');
}

TEST(LlvmLibcWctypeUtilsTest, IsLowerAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::islower;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = c >= 'a' && c <= 'z';
    EXPECT_EQ(islower(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x03B1, "GREEK SMALL LETTER ALPHA", false},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(islower(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsLowerUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::islower;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = c >= 'a' && c <= 'z';
    EXPECT_EQ(islower(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", true},
      {0x03B1, "GREEK SMALL LETTER ALPHA", true},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(islower(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsUpperAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isupper;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = c >= 'A' && c <= 'Z';
    EXPECT_EQ(isupper(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x03B1, "GREEK SMALL LETTER ALPHA", false},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isupper(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsUpperUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isupper;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = c >= 'A' && c <= 'Z';
    EXPECT_EQ(isupper(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x03B1, "GREEK SMALL LETTER ALPHA", false},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isupper(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsAlphaAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isalpha;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    EXPECT_EQ(isalpha(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x03B1, "GREEK SMALL LETTER ALPHA", false},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isalpha(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsAlphaUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isalpha;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    EXPECT_EQ(isalpha(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", true},
      {0x03B1, "GREEK SMALL LETTER ALPHA", true},
      {0x00C0, "LATIN CAPITAL LETTER A WITH GRAVE", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isalpha(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsDigitAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isdigit;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= '0' && c <= '9');
    EXPECT_EQ(isdigit(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x0660, "ARABIC-INDIC DIGIT ZERO", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isdigit(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsDigitUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isdigit;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= '0' && c <= '9');
    EXPECT_EQ(isdigit(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  // In C.UTF-8, isdigit only returns true for ASCII digits.
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x0660, "ARABIC-INDIC DIGIT ZERO", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isdigit(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsAlnumAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isalnum;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z');
    EXPECT_EQ(isalnum(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", false},
      {0x0660, "ARABIC-INDIC DIGIT ZERO", false},
      {0x0030, "DIGIT ZERO", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isalnum(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsAlnumUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isalnum;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z');
    EXPECT_EQ(isalnum(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00E9, "LATIN SMALL LETTER E WITH ACUTE", true},
      {0x0660, "ARABIC-INDIC DIGIT ZERO", true},
      {0x0030, "DIGIT ZERO", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isalnum(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsXDigitAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isxdigit;

  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                    (c >= 'A' && c <= 'F');
    EXPECT_EQ(isxdigit(static_cast<wchar_t>(c)), expected);
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsXDigitUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isxdigit;

  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                    (c >= 'A' && c <= 'F');
    EXPECT_EQ(isxdigit(static_cast<wchar_t>(c)), expected);
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsSpaceAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isspace;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c == ' ' || c == '\t' || c == '\n' || c == '\v' ||
                     c == '\f' || c == '\r');
    EXPECT_EQ(isspace(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", false},
      {0x2000, "EN QUAD", false},
      {0x2028, "LINE SEPARATOR", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isspace(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsSpaceUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isspace;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c == ' ' || c == '\t' || c == '\n' || c == '\v' ||
                     c == '\f' || c == '\r');
    EXPECT_EQ(isspace(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", false},
      {0x1680, "OGHAM SPACE MARK", true},
      {0x2000, "EN QUAD", true},
      {0x2028, "LINE SEPARATOR", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isspace(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsBlankAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isblank;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c == ' ' || c == '\t');
    EXPECT_EQ(isblank(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", false},
      {0x2000, "EN QUAD", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isblank(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsBlankUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isblank;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c == ' ' || c == '\t');
    EXPECT_EQ(isblank(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", false},
      {0x1680, "OGHAM SPACE MARK", true},
      {0x2000, "EN QUAD", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isblank(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsGraphAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isgraph;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c > 0x20 && c < 0x7f);
    EXPECT_EQ(isgraph(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", false},
      {0x2000, "EN QUAD", false},
      {0x2603, "SNOWMAN", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isgraph(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsGraphUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isgraph;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c > 0x20 && c < 0x7f);
    EXPECT_EQ(isgraph(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", true},
      {0x2000, "EN QUAD", false},
      {0x2603, "SNOWMAN", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isgraph(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsPrintAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isprint;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= 0x20 && c < 0x7f);
    EXPECT_EQ(isprint(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", false},
      {0x2000, "EN QUAD", false},
      {0x2603, "SNOWMAN", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isprint(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsPrintUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isprint;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c >= 0x20 && c < 0x7f);
    EXPECT_EQ(isprint(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A0, "NO-BREAK SPACE", true},
      {0x2000, "EN QUAD", true},
      {0x2603, "SNOWMAN", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(isprint(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsPunctAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::isalnum;
  using ascii_mode::LIBC_NAMESPACE::internal::isgraph;
  using ascii_mode::LIBC_NAMESPACE::internal::ispunct;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = is_punctuation_character(c);
    EXPECT_EQ(ispunct(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A1, "INVERTED EXCLAMATION MARK", false},
      {0x2014, "EM DASH", false},
      {0x20AC, "EURO SIGN", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(ispunct(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsPunctUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::isalnum;
  using utf8_mode::LIBC_NAMESPACE::internal::isgraph;
  using utf8_mode::LIBC_NAMESPACE::internal::ispunct;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = is_punctuation_character(c);
    EXPECT_EQ(ispunct(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x00A1, "INVERTED EXCLAMATION MARK", true},
      {0x2014, "EM DASH", true},
      {0x20AC, "EURO SIGN", true},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(ispunct(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsCntrlAscii) {
  using ascii_mode::LIBC_NAMESPACE::internal::iscntrl;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c < 0x20 || c == 0x7f);
    EXPECT_EQ(iscntrl(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x0080, "PADDING CHARACTER", false},
      {0x009F, "APPLICATION PROGRAM COMMAND", false},
      {0x2028, "LINE SEPARATOR", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(iscntrl(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

TEST(LlvmLibcWctypeUtilsTest, IsCntrlUtf8) {
  using utf8_mode::LIBC_NAMESPACE::internal::iscntrl;

  // ASCII
  for (int c = 0; c < 128; ++c) {
    bool expected = (c < 0x20 || c == 0x7f);
    EXPECT_EQ(iscntrl(static_cast<wchar_t>(c)), expected);
  }

  // Non ASCII
  TestCase cases[] = {
      {0x0080, "PADDING CHARACTER", true},
      {0x009F, "APPLICATION PROGRAM COMMAND", true},
      {0x2028, "LINE SEPARATOR", false},
  };

  for (const auto &tc : cases) {
    EXPECT_EQ(iscntrl(static_cast<wchar_t>(tc.wc)), tc.expected) << tc.name;
  }
}

} // namespace
