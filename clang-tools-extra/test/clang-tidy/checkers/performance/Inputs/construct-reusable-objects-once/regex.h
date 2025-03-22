#ifndef _REGEX_
#define _REGEX_

namespace std {

template <typename T>
class regex_traits {};

namespace regex_constants {
  enum syntax_option_type : unsigned int {
    _S_icase		= 1 << 0,
    _S_nosubs		= 1 << 1,
    _S_basic		= 1 << 2,
    _S_ECMAScript = 1 << 3,
  };
} // namespace regex_constants

template<class CharT, class Traits = std::regex_traits<CharT>>
class basic_regex {
public:
  typedef regex_constants::syntax_option_type flag_type;

  static constexpr flag_type icase = regex_constants::syntax_option_type::_S_icase;
  static constexpr flag_type nosubs = regex_constants::syntax_option_type::_S_nosubs;
  static constexpr flag_type basic = regex_constants::syntax_option_type::_S_basic;
  static constexpr flag_type ECMAScript = regex_constants::syntax_option_type::_S_ECMAScript;

  basic_regex();
  explicit basic_regex(const CharT* s, flag_type f = ECMAScript);
  basic_regex(const CharT* s, unsigned int count, std::regex_constants::syntax_option_type f = ECMAScript);
  basic_regex(const basic_regex& other);
  basic_regex(basic_regex&& other) noexcept;
  template<class ForwardIt>
  basic_regex(ForwardIt first, ForwardIt last, std::regex_constants::syntax_option_type f = ECMAScript);

  basic_regex& operator=(const basic_regex& other);
  basic_regex& operator=(basic_regex&& other) noexcept;
};

typedef basic_regex<char> regex;
typedef basic_regex<wchar_t> wregex;

} // namespace std

#endif // _REGEX_
