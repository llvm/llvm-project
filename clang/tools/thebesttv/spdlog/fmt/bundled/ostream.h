// Formatting library for C++ - std::ostream support
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_OSTREAM_H_
#define FMT_OSTREAM_H_

#include <fstream>  // std::filebuf

#ifdef _WIN32
#  ifdef __GLIBCXX__
#    include <ext/stdio_filebuf.h>
#    include <ext/stdio_sync_filebuf.h>
#  endif
#  include <io.h>
#endif

#include "format.h"

FMT_BEGIN_NAMESPACE
namespace detail {

template <typename Streambuf> class formatbuf : public Streambuf {
 private:
  using char_type = typename Streambuf::char_type;
  using streamsize = decltype(std::declval<Streambuf>().sputn(nullptr, 0));
  using int_type = typename Streambuf::int_type;
  using traits_type = typename Streambuf::traits_type;

  buffer<char_type>& buffer_;

 public:
  explicit formatbuf(buffer<char_type>& buf) : buffer_(buf) {}

 protected:
  // The put area is always empty. This makes the implementation simpler and has
  // the advantage that the streambuf and the buffer are always in sync and
  // sputc never writes into uninitialized memory. A disadvantage is that each
  // call to sputc always results in a (virtual) call to overflow. There is no
  // disadvantage here for sputn since this always results in a call to xsputn.

  auto overflow(int_type ch) -> int_type override {
    if (!traits_type::eq_int_type(ch, traits_type::eof()))
      buffer_.push_back(static_cast<char_type>(ch));
    return ch;
  }

  auto xsputn(const char_type* s, streamsize count) -> streamsize override {
    buffer_.append(s, s + count);
    return count;
  }
};

// Generate a unique explicit instantion in every translation unit using a tag
// type in an anonymous namespace.
namespace {
struct file_access_tag {};
}  // namespace
template <typename Tag, typename BufType, FILE* BufType::*FileMemberPtr>
class file_access {
  friend auto get_file(BufType& obj) -> FILE* { return obj.*FileMemberPtr; }
};

#if FMT_MSC_VERSION
template class file_access<file_access_tag, std::filebuf,
                           &std::filebuf::_Myfile>;
auto get_file(std::filebuf&) -> FILE*;
#endif

inline auto write_ostream_unicode(std::ostream& os, fmt::string_view data)
    -> bool {
  FILE* f = nullptr;
#if FMT_MSC_VERSION
  if (auto* buf = dynamic_cast<std::filebuf*>(os.rdbuf()))
    f = get_file(*buf);
  else
    return false;
#elif defined(_WIN32) && defined(__GLIBCXX__)
  auto* rdbuf = os.rdbuf();
  if (auto* sfbuf = dynamic_cast<__gnu_cxx::stdio_sync_filebuf<char>*>(rdbuf))
    f = sfbuf->file();
  else if (auto* fbuf = dynamic_cast<__gnu_cxx::stdio_filebuf<char>*>(rdbuf))
    f = fbuf->file();
  else
    return false;
#else
  ignore_unused(os, data, f);
#endif
#ifdef _WIN32
  if (f) {
    int fd = _fileno(f);
    if (_isatty(fd)) {
      os.flush();
      return write_console(fd, data);
    }
  }
#endif
  return false;
}
inline auto write_ostream_unicode(std::wostream&,
                                  fmt::basic_string_view<wchar_t>) -> bool {
  return false;
}

// Write the content of buf to os.
// It is a separate function rather than a part of vprint to simplify testing.
template <typename Char>
void write_buffer(std::basic_ostream<Char>& os, buffer<Char>& buf) {
  const Char* buf_data = buf.data();
  using unsigned_streamsize = std::make_unsigned<std::streamsize>::type;
  unsigned_streamsize size = buf.size();
  unsigned_streamsize max_size = to_unsigned(max_value<std::streamsize>());
  do {
    unsigned_streamsize n = size <= max_size ? size : max_size;
    os.write(buf_data, static_cast<std::streamsize>(n));
    buf_data += n;
    size -= n;
  } while (size != 0);
}

template <typename Char, typename T>
void format_value(buffer<Char>& buf, const T& value) {
  auto&& format_buf = formatbuf<std::basic_streambuf<Char>>(buf);
  auto&& output = std::basic_ostream<Char>(&format_buf);
#if !defined(FMT_STATIC_THOUSANDS_SEPARATOR)
  output.imbue(std::locale::classic());  // The default is always unlocalized.
#endif
  output << value;
  output.exceptions(std::ios_base::failbit | std::ios_base::badbit);
}

template <typename T> struct streamed_view {
  const T& value;
};

}  // namespace detail

// Formats an object of type T that has an overloaded ostream operator<<.
template <typename Char>
struct basic_ostream_formatter : formatter<basic_string_view<Char>, Char> {
  void set_debug_format() = delete;

  template <typename T, typename OutputIt>
  auto format(const T& value, basic_format_context<OutputIt, Char>& ctx) const
      -> OutputIt {
    auto buffer = basic_memory_buffer<Char>();
    detail::format_value(buffer, value);
    return formatter<basic_string_view<Char>, Char>::format(
        {buffer.data(), buffer.size()}, ctx);
  }
};

using ostream_formatter = basic_ostream_formatter<char>;

template <typename T, typename Char>
struct formatter<detail::streamed_view<T>, Char>
    : basic_ostream_formatter<Char> {
  template <typename OutputIt>
  auto format(detail::streamed_view<T> view,
              basic_format_context<OutputIt, Char>& ctx) const -> OutputIt {
    return basic_ostream_formatter<Char>::format(view.value, ctx);
  }
};

/**
  \rst
  Returns a view that formats `value` via an ostream ``operator<<``.

  **Example**::

    fmt::print("Current thread id: {}\n",
               fmt::streamed(std::this_thread::get_id()));
  \endrst
 */
template <typename T>
constexpr auto streamed(const T& value) -> detail::streamed_view<T> {
  return {value};
}

namespace detail {

inline void vprint_directly(std::ostream& os, string_view format_str,
                            format_args args) {
  auto buffer = memory_buffer();
  detail::vformat_to(buffer, format_str, args);
  detail::write_buffer(os, buffer);
}

}  // namespace detail

FMT_EXPORT template <typename Char>
void vprint(std::basic_ostream<Char>& os,
            basic_string_view<type_identity_t<Char>> format_str,
            basic_format_args<buffer_context<type_identity_t<Char>>> args) {
  auto buffer = basic_memory_buffer<Char>();
  detail::vformat_to(buffer, format_str, args);
  if (detail::write_ostream_unicode(os, {buffer.data(), buffer.size()})) return;
  detail::write_buffer(os, buffer);
}

/**
  \rst
  Prints formatted data to the stream *os*.

  **Example**::

    fmt::print(cerr, "Don't {}!", "panic");
  \endrst
 */
FMT_EXPORT template <typename... T>
void print(std::ostream& os, format_string<T...> fmt, T&&... args) {
  const auto& vargs = fmt::make_format_args(args...);
  if (detail::is_utf8())
    vprint(os, fmt, vargs);
  else
    detail::vprint_directly(os, fmt, vargs);
}

FMT_EXPORT
template <typename... Args>
void print(std::wostream& os,
           basic_format_string<wchar_t, type_identity_t<Args>...> fmt,
           Args&&... args) {
  vprint(os, fmt, fmt::make_format_args<buffer_context<wchar_t>>(args...));
}

FMT_EXPORT template <typename... T>
void println(std::ostream& os, format_string<T...> fmt, T&&... args) {
  fmt::print(os, "{}\n", fmt::format(fmt, std::forward<T>(args)...));
}

FMT_EXPORT
template <typename... Args>
void println(std::wostream& os,
             basic_format_string<wchar_t, type_identity_t<Args>...> fmt,
             Args&&... args) {
  print(os, L"{}\n", fmt::format(fmt, std::forward<Args>(args)...));
}

FMT_END_NAMESPACE

#endif  // FMT_OSTREAM_H_
