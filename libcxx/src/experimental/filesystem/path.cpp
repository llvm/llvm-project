//===--------------------- filesystem/path.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "experimental/filesystem"
#include "string_view"
#include "utility"

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_FILESYSTEM

_LIBCPP_CONSTEXPR path::value_type path::preferred_separator;


using string_view_t = path::__string_view;

namespace { namespace parser
{

using value_type = path::value_type;
using string_view_pair = pair<string_view_t, string_view_t>;

// status reporting
constexpr size_t npos = static_cast<size_t>(-1);

inline bool good(size_t pos) { return pos != npos; }

// lexical elements
constexpr value_type preferred_separator = path::preferred_separator;
constexpr value_type const * preferred_separator_str = "/";
constexpr value_type const * dot = ".";

// forward //
bool is_separator(string_view_t const &, size_t);
bool is_root_name(const string_view_t&, size_t);
bool is_root_directory(string_view_t const &, size_t);
bool is_trailing_separator(string_view_t const &, size_t);

size_t start_of(string_view_t const &, size_t);
size_t end_of(string_view_t const &, size_t);

size_t root_name_start(const string_view_t& s);
size_t root_name_end(const string_view_t&);

size_t root_directory_start(string_view_t const &);
size_t root_directory_end(string_view_t const &);

string_view_pair separate_filename(string_view_t const &);
string_view extract_raw(string_view_t const &, size_t);
string_view extract_preferred(string_view_t const &, size_t);

inline bool is_separator(const string_view_t& s, size_t pos) {
    return (pos < s.size() && s[pos] == preferred_separator);
}

inline bool is_root_name(const string_view_t& s, size_t pos) {
  return good(pos) && pos == 0 ? root_name_start(s) == pos : false;
}

inline bool is_root_directory(const string_view_t& s, size_t pos) {
    return good(pos) ? root_directory_start(s) == pos : false;
}

inline bool is_trailing_separator(const string_view_t& s, size_t pos) {
    return (pos < s.size() && is_separator(s, pos) &&
            end_of(s, pos) == s.size()-1 &&
            !is_root_directory(s, pos) && !is_root_name(s, pos));
}

size_t start_of(const string_view_t& s, size_t pos) {
    if (pos >= s.size()) return npos;
    bool in_sep = (s[pos] == preferred_separator);
    while (pos - 1 < s.size() &&
        (s[pos-1] == preferred_separator) == in_sep)
    { --pos; }
    if (pos == 2 && !in_sep && s[0] == preferred_separator &&
        s[1] == preferred_separator)
    { return 0; }
    return pos;
}

size_t end_of(const string_view_t& s, size_t pos) {
    if (pos >= s.size()) return npos;
    // special case for root name
    if (pos == 0 && is_root_name(s, pos)) return root_name_end(s);
    bool in_sep = (s[pos] == preferred_separator);
    while (pos + 1 < s.size() && (s[pos+1] == preferred_separator) == in_sep)
    { ++pos; }
    return pos;
}

inline size_t root_name_start(const string_view_t& s) {
    return good(root_name_end(s)) ? 0 : npos;
}

size_t root_name_end(const string_view_t& s) {
    if (s.size() < 2 || s[0] != preferred_separator
        || s[1] != preferred_separator) {
        return npos;
    }
    if (s.size() == 2) {
        return 1;
    }
    size_t index = 2; // current position
    if (s[index] == preferred_separator) {
        return npos;
    }
    while (index + 1 < s.size() && s[index+1] != preferred_separator) {
        ++index;
    }
    return index;
}

size_t root_directory_start(const string_view_t& s) {
    size_t e = root_name_end(s);
    if (!good(e))
    return is_separator(s, 0) ? 0 : npos;
    return is_separator(s, e + 1) ? e + 1 : npos;
}

size_t root_directory_end(const string_view_t& s) {
    size_t st = root_directory_start(s);
    if (!good(st)) return npos;
    size_t index = st;
    while (index + 1 < s.size() && s[index + 1] == preferred_separator)
      { ++index; }
    return index;
}

string_view_pair separate_filename(string_view_t const & s) {
    if (s == "." || s == ".." || s.empty()) return string_view_pair{s, ""};
    auto pos = s.find_last_of('.');
    if (pos == string_view_t::npos) return string_view_pair{s, string_view{}};
    return string_view_pair{s.substr(0, pos), s.substr(pos)};
}

inline string_view extract_raw(const string_view_t& s, size_t pos) {
    size_t end_i = end_of(s, pos);
    if (!good(end_i)) return string_view{};
    return string_view(s).substr(pos, end_i - pos + 1);
}

string_view extract_preferred(const string_view_t& s, size_t pos) {
    string_view raw = extract_raw(s, pos);
    if (raw.empty())
        return raw;
    if (is_trailing_separator(s, pos))
        return string_view{dot};
    if (is_separator(s, pos) && !is_root_name(s, pos))
        return string_view(preferred_separator_str);
    return raw;
}

}} // namespace parser


////////////////////////////////////////////////////////////////////////////////
//                            path_view_iterator
////////////////////////////////////////////////////////////////////////////////
namespace {

struct path_view_iterator {
  const string_view __s_;
  size_t __pos_;

  explicit path_view_iterator(string_view const& __s) : __s_(__s), __pos_(__s_.empty() ? parser::npos : 0) {}
  explicit path_view_iterator(string_view const& __s, size_t __p) : __s_(__s), __pos_(__p) {}

  string_view operator*() const {
    return parser::extract_preferred(__s_, __pos_);
  }

  path_view_iterator& operator++() {
    increment();
    return *this;
  }

  path_view_iterator& operator--() {
    decrement();
    return *this;
  }

  void increment() {
    if (__pos_ == parser::npos) return;
    while (! set_position(parser::end_of(__s_, __pos_)+1))
        ;
    return;
  }

  void decrement() {
    if (__pos_ == 0) {
      set_position(0);
    }
    else if (__pos_ == parser::npos) {
      auto const str_size = __s_.size();
      set_position(parser::start_of(
          __s_, str_size != 0 ? str_size - 1 : str_size));
    } else {
      while (!set_position(parser::start_of(__s_, __pos_-1)))
        ;
    }
  }

  bool set_position(size_t pos) {
    if (pos >= __s_.size()) {
      __pos_ = parser::npos;
    } else {
      __pos_ = pos;
    }
    return valid_iterator_position();
  }

  bool valid_iterator_position() const {
    if (__pos_ == parser::npos) return true; // end position is valid
    return (!parser::is_separator      (__s_, __pos_) ||
          parser::is_root_directory    (__s_, __pos_) ||
          parser::is_trailing_separator(__s_, __pos_) ||
          parser::is_root_name         (__s_, __pos_));
  }

  bool is_end() const { return __pos_ == parser::npos; }

  inline bool operator==(path_view_iterator const& __p) {
      return __pos_ == __p.__pos_;
  }
};

path_view_iterator pbegin(path const& p) {
    return path_view_iterator(p.native());
}

path_view_iterator pend(path const& p) {
    path_view_iterator __p(p.native());
    __p.__pos_ = parser::npos;
    return __p;
}

} // end namespace
///////////////////////////////////////////////////////////////////////////////
//                            path definitions
///////////////////////////////////////////////////////////////////////////////

path & path::replace_extension(path const & replacement)
{
    path p = extension();
    if (not p.empty()) {
      __pn_.erase(__pn_.size() - p.native().size());
    }
    if (!replacement.empty()) {
        if (replacement.native()[0] != '.') {
            __pn_ += ".";
        }
        __pn_.append(replacement.__pn_);
    }
    return *this;
}

///////////////////////////////////////////////////////////////////////////////
// path.decompose

string_view_t path::__root_name() const
{
    return parser::is_root_name(__pn_, 0)
      ? parser::extract_preferred(__pn_, 0)
      : string_view_t{};
}

string_view_t path::__root_directory() const
{
    auto start_i = parser::root_directory_start(__pn_);
    if(!parser::good(start_i)) {
        return {};
    }
    return parser::extract_preferred(__pn_, start_i);
}

string_view_t path::__root_path_raw() const
{
    size_t e = parser::root_directory_end(__pn_);
    if (!parser::good(e))
      e = parser::root_name_end(__pn_);
    if (parser::good(e))
      return string_view_t(__pn_).substr(0, e + 1);
    return {};
}

string_view_t path::__relative_path() const
{
    if (empty()) {
        return __pn_;
    }
    auto end_i = parser::root_directory_end(__pn_);
    if (not parser::good(end_i)) {
        end_i = parser::root_name_end(__pn_);
    }
    if (not parser::good(end_i)) {
        return __pn_;
    }
    return string_view_t(__pn_).substr(end_i+1);
}

string_view_t path::__parent_path() const
{
    if (empty() || pbegin(*this) == --pend(*this)) {
        return {};
    }
    auto end_it = --(--pend(*this));
    auto end_i = parser::end_of(__pn_, end_it.__pos_);
    return string_view_t(__pn_).substr(0, end_i+1);
}

string_view_t path::__filename() const
{
    return empty() ? string_view_t{} : *--pend(*this);
}

string_view_t path::__stem() const
{
    return parser::separate_filename(__filename()).first;
}

string_view_t path::__extension() const
{
    return parser::separate_filename(__filename()).second;
}

////////////////////////////////////////////////////////////////////////////
// path.comparisons
int path::__compare(string_view_t __s) const {
    path_view_iterator thisIter(this->native());
    path_view_iterator sIter(__s);
    while (!thisIter.is_end() && !sIter.is_end()) {
        int res = (*thisIter).compare(*sIter);
        if (res != 0) return res;
        ++thisIter; ++sIter;
    }
    if (thisIter.is_end() && sIter.is_end())
        return 0;
    if (thisIter.is_end())
        return -1;
    return 1;
}

////////////////////////////////////////////////////////////////////////////
// path.nonmembers
size_t hash_value(const path& __p) _NOEXCEPT {
  path_view_iterator thisIter(__p.native());
  struct HashPairT {
    size_t first;
    size_t second;
  };
  HashPairT hp = {0, 0};
  std::hash<string_view> hasher;
  std::__scalar_hash<decltype(hp)> pair_hasher;
  while (!thisIter.is_end()) {
    hp.second = hasher(*thisIter);
    hp.first = pair_hasher(hp);
    ++thisIter;
  }
  return hp.first;
}

////////////////////////////////////////////////////////////////////////////
// path.itr
path::iterator path::begin() const
{
    path_view_iterator pit = pbegin(*this);
    iterator it;
    it.__path_ptr_ = this;
    it.__pos_ = pit.__pos_;
    it.__elem_.__assign_view(*pit);
    return it;
}

path::iterator path::end() const
{
    iterator it{};
    it.__path_ptr_ = this;
    it.__pos_ = parser::npos;
    return it;
}

path::iterator& path::iterator::__increment() {
  path_view_iterator it(__path_ptr_->native(), __pos_);
  it.increment();
  __pos_ = it.__pos_;
  __elem_.__assign_view(*it);
  return *this;
}

path::iterator& path::iterator::__decrement() {
  path_view_iterator it(__path_ptr_->native(), __pos_);
  it.decrement();
  __pos_ = it.__pos_;
  __elem_.__assign_view(*it);
  return *this;
}

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_FILESYSTEM
