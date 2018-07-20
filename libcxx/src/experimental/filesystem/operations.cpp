//===--------------------- filesystem/ops.cpp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "experimental/filesystem"
#include "iterator"
#include "fstream"
#include "random"  /* for unique_path */
#include "string_view"
#include "type_traits"
#include "vector"
#include "cstdlib"
#include "climits"

#include "filesystem_common.h"

#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>  /* values for fchmodat */
#include <experimental/filesystem>

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_FILESYSTEM

filesystem_error::~filesystem_error() {}


namespace { namespace parser
{

using string_view_t = path::__string_view;
using string_view_pair = pair<string_view_t, string_view_t>;
using PosPtr = path::value_type const*;

struct PathParser {
  enum ParserState : unsigned char {
    // Zero is a special sentinel value used by default constructed iterators.
    PS_BeforeBegin = 1,
    PS_InRootName,
    PS_InRootDir,
    PS_InFilenames,
    PS_InTrailingSep,
    PS_AtEnd
  };

  const string_view_t Path;
  string_view_t RawEntry;
  ParserState State;

private:
  PathParser(string_view_t P, ParserState State) noexcept
      : Path(P), State(State) {}

public:
  PathParser(string_view_t P, string_view_t E, unsigned char S)
      : Path(P), RawEntry(E), State(static_cast<ParserState>(S)) {
    // S cannot be '0' or PS_BeforeBegin.
  }

  static PathParser CreateBegin(string_view_t P) noexcept {
    PathParser PP(P, PS_BeforeBegin);
    PP.increment();
    return PP;
  }

  static PathParser CreateEnd(string_view_t P) noexcept {
    PathParser PP(P, PS_AtEnd);
    return PP;
  }

  PosPtr peek() const noexcept {
    auto TkEnd = getNextTokenStartPos();
    auto End = getAfterBack();
    return TkEnd == End ? nullptr : TkEnd;
  }

  void increment() noexcept {
    const PosPtr End = getAfterBack();
    const PosPtr Start = getNextTokenStartPos();
    if (Start == End)
      return makeState(PS_AtEnd);

    switch (State) {
    case PS_BeforeBegin: {
      PosPtr TkEnd = consumeSeparator(Start, End);
      if (TkEnd)
        return makeState(PS_InRootDir, Start, TkEnd);
      else
        return makeState(PS_InFilenames, Start, consumeName(Start, End));
    }
    case PS_InRootDir:
      return makeState(PS_InFilenames, Start, consumeName(Start, End));

    case PS_InFilenames: {
      PosPtr SepEnd = consumeSeparator(Start, End);
      if (SepEnd != End) {
        PosPtr TkEnd = consumeName(SepEnd, End);
        if (TkEnd)
          return makeState(PS_InFilenames, SepEnd, TkEnd);
      }
      return makeState(PS_InTrailingSep, Start, SepEnd);
    }

    case PS_InTrailingSep:
      return makeState(PS_AtEnd);

    case PS_InRootName:
    case PS_AtEnd:
      _LIBCPP_UNREACHABLE();
    }
  }

  void decrement() noexcept {
    const PosPtr REnd = getBeforeFront();
    const PosPtr RStart = getCurrentTokenStartPos() - 1;
    if (RStart == REnd) // we're decrementing the begin
      return makeState(PS_BeforeBegin);

    switch (State) {
    case PS_AtEnd: {
      // Try to consume a trailing separator or root directory first.
      if (PosPtr SepEnd = consumeSeparator(RStart, REnd)) {
        if (SepEnd == REnd)
          return makeState(PS_InRootDir, Path.data(), RStart + 1);
        return makeState(PS_InTrailingSep, SepEnd + 1, RStart + 1);
      } else {
        PosPtr TkStart = consumeName(RStart, REnd);
        return makeState(PS_InFilenames, TkStart + 1, RStart + 1);
      }
    }
    case PS_InTrailingSep:
      return makeState(PS_InFilenames, consumeName(RStart, REnd) + 1, RStart + 1);
    case PS_InFilenames: {
      PosPtr SepEnd = consumeSeparator(RStart, REnd);
      if (SepEnd == REnd)
        return makeState(PS_InRootDir, Path.data(), RStart + 1);
      PosPtr TkEnd = consumeName(SepEnd, REnd);
      return makeState(PS_InFilenames, TkEnd + 1, SepEnd + 1);
    }
    case PS_InRootDir:
      // return makeState(PS_InRootName, Path.data(), RStart + 1);
    case PS_InRootName:
    case PS_BeforeBegin:
      _LIBCPP_UNREACHABLE();
    }
  }

  /// \brief Return a view with the "preferred representation" of the current
  ///   element. For example trailing separators are represented as a '.'
  string_view_t operator*() const noexcept {
    switch (State) {
    case PS_BeforeBegin:
    case PS_AtEnd:
      return "";
    case PS_InRootDir:
      return "/";
    case PS_InTrailingSep:
      return "";
    case PS_InRootName:
    case PS_InFilenames:
      return RawEntry;
    }
    _LIBCPP_UNREACHABLE();
  }

  explicit operator bool() const noexcept {
    return State != PS_BeforeBegin && State != PS_AtEnd;
  }

  PathParser& operator++() noexcept {
    increment();
    return *this;
  }

  PathParser& operator--() noexcept {
    decrement();
    return *this;
  }

  bool inRootPath() const noexcept {
    return State == PS_InRootDir || State == PS_InRootName;
  }

private:
  void makeState(ParserState NewState, PosPtr Start, PosPtr End) noexcept {
    State = NewState;
    RawEntry = string_view_t(Start, End - Start);
  }
  void makeState(ParserState NewState) noexcept {
    State = NewState;
    RawEntry = {};
  }

  PosPtr getAfterBack() const noexcept {
    return Path.data() + Path.size();
  }

  PosPtr getBeforeFront() const noexcept {
    return Path.data() - 1;
  }

  /// \brief Return a pointer to the first character after the currently
  ///   lexed element.
  PosPtr getNextTokenStartPos() const noexcept {
    switch (State) {
    case PS_BeforeBegin:
      return Path.data();
    case PS_InRootName:
    case PS_InRootDir:
    case PS_InFilenames:
      return &RawEntry.back() + 1;
    case PS_InTrailingSep:
    case PS_AtEnd:
      return getAfterBack();
    }
    _LIBCPP_UNREACHABLE();
  }

  /// \brief Return a pointer to the first character in the currently lexed
  ///   element.
  PosPtr getCurrentTokenStartPos() const noexcept {
    switch (State) {
    case PS_BeforeBegin:
    case PS_InRootName:
      return &Path.front();
    case PS_InRootDir:
    case PS_InFilenames:
    case PS_InTrailingSep:
      return &RawEntry.front();
    case PS_AtEnd:
      return &Path.back() + 1;
    }
    _LIBCPP_UNREACHABLE();
  }

  PosPtr consumeSeparator(PosPtr P, PosPtr End) const noexcept {
    if (P == End || *P != '/')
      return nullptr;
    const int Inc = P < End ? 1 : -1;
    P += Inc;
    while (P != End && *P == '/')
      P += Inc;
    return P;
  }

  PosPtr consumeName(PosPtr P, PosPtr End) const noexcept {
    if (P == End || *P == '/')
      return nullptr;
    const int Inc = P < End ? 1 : -1;
    P += Inc;
    while (P != End && *P != '/')
      P += Inc;
    return P;
  }
};

string_view_pair separate_filename(string_view_t const & s) {
    if (s == "." || s == ".." || s.empty()) return string_view_pair{s, ""};
    auto pos = s.find_last_of('.');
    if (pos == string_view_t::npos || pos == 0)
        return string_view_pair{s, string_view_t{}};
    return string_view_pair{s.substr(0, pos), s.substr(pos)};
}

string_view_t createView(PosPtr S, PosPtr E) noexcept {
  return {S, static_cast<size_t>(E - S) + 1};
}

}} // namespace parser


//                       POSIX HELPERS

namespace detail { namespace  {

using value_type = path::value_type;
using string_type = path::string_type;

perms posix_get_perms(const struct ::stat& st) noexcept {
  return static_cast<perms>(st.st_mode) & perms::mask;
}

::mode_t posix_convert_perms(perms prms) {
  return static_cast< ::mode_t>(prms & perms::mask);
}

file_status create_file_status(std::error_code& m_ec, path const& p,
                               struct ::stat& path_stat, std::error_code* ec) {
  if (ec)
    *ec = m_ec;
  if (m_ec && (m_ec.value() == ENOENT || m_ec.value() == ENOTDIR)) {
    return file_status(file_type::not_found);
  } else if (m_ec) {
    set_or_throw(m_ec, ec, "posix_stat", p);
    return file_status(file_type::none);
  }
  // else

  file_status fs_tmp;
  auto const mode = path_stat.st_mode;
  if (S_ISLNK(mode))
    fs_tmp.type(file_type::symlink);
  else if (S_ISREG(mode))
    fs_tmp.type(file_type::regular);
  else if (S_ISDIR(mode))
    fs_tmp.type(file_type::directory);
  else if (S_ISBLK(mode))
    fs_tmp.type(file_type::block);
  else if (S_ISCHR(mode))
    fs_tmp.type(file_type::character);
  else if (S_ISFIFO(mode))
    fs_tmp.type(file_type::fifo);
  else if (S_ISSOCK(mode))
    fs_tmp.type(file_type::socket);
  else
    fs_tmp.type(file_type::unknown);

  fs_tmp.permissions(detail::posix_get_perms(path_stat));
  return fs_tmp;
}

file_status posix_stat(path const& p, struct ::stat& path_stat,
                       std::error_code* ec) {
  std::error_code m_ec;
  if (::stat(p.c_str(), &path_stat) == -1)
    m_ec = detail::capture_errno();
  return create_file_status(m_ec, p, path_stat, ec);
}

file_status posix_stat(path const& p, std::error_code* ec) {
  struct ::stat path_stat;
  return posix_stat(p, path_stat, ec);
}

file_status posix_lstat(path const& p, struct ::stat& path_stat,
                        std::error_code* ec) {
  std::error_code m_ec;
  if (::lstat(p.c_str(), &path_stat) == -1)
    m_ec = detail::capture_errno();
  return create_file_status(m_ec, p, path_stat, ec);
}

file_status posix_lstat(path const& p, std::error_code* ec) {
  struct ::stat path_stat;
  return posix_lstat(p, path_stat, ec);
}

bool stat_equivalent(struct ::stat& st1, struct ::stat& st2) {
  return (st1.st_dev == st2.st_dev && st1.st_ino == st2.st_ino);
}

//                           DETAIL::MISC


bool copy_file_impl(const path& from, const path& to, perms from_perms,
                    std::error_code *ec)
{
    std::ifstream in(from.c_str(), std::ios::binary);
    std::ofstream out(to.c_str(),  std::ios::binary);

    if (in.good() && out.good()) {
        using InIt = std::istreambuf_iterator<char>;
        using OutIt = std::ostreambuf_iterator<char>;
        InIt bin(in);
        InIt ein;
        OutIt bout(out);
        std::copy(bin, ein, bout);
    }
    if (out.fail() || in.fail()) {
        set_or_throw(make_error_code(errc::operation_not_permitted),
                     ec, "copy_file", from, to);
        return false;
    }
    __permissions(to, from_perms, perm_options::replace, ec);
    // TODO what if permissions fails?
    return true;
}

}} // end namespace detail

using detail::set_or_throw;
using parser::string_view_t;
using parser::PathParser;
using parser::createView;

static path __do_absolute(const path& p, path *cwd, std::error_code *ec) {
  if (ec) ec->clear();
    if (p.is_absolute())
      return p;
    *cwd = __current_path(ec);
    if (ec && *ec)
      return {};
    return (*cwd) / p;
}

path __absolute(const path& p, std::error_code *ec) {
    path cwd;
    return __do_absolute(p, &cwd, ec);
}

path __canonical(path const & orig_p, std::error_code *ec)
{
    path cwd;
    path p = __do_absolute(orig_p, &cwd, ec);
    char buff[PATH_MAX + 1];
    char *ret;
    if ((ret = ::realpath(p.c_str(), buff)) == nullptr) {
        set_or_throw(ec, "canonical", orig_p, cwd);
        return {};
    }
    if (ec) ec->clear();
    return {ret};
}

void __copy(const path& from, const path& to, copy_options options,
            std::error_code *ec)
{
    const bool sym_status = bool(options &
        (copy_options::create_symlinks | copy_options::skip_symlinks));

    const bool sym_status2 = bool(options &
        copy_options::copy_symlinks);

    std::error_code m_ec1;
    struct ::stat f_st = {};
    const file_status f = sym_status || sym_status2
                                     ? detail::posix_lstat(from, f_st, &m_ec1)
                                     : detail::posix_stat(from,  f_st, &m_ec1);
    if (m_ec1)
        return set_or_throw(m_ec1, ec, "copy", from, to);

    struct ::stat t_st = {};
    const file_status t = sym_status ? detail::posix_lstat(to, t_st, &m_ec1)
                                     : detail::posix_stat(to, t_st, &m_ec1);

    if (not status_known(t))
        return set_or_throw(m_ec1, ec, "copy", from, to);

    if (!exists(f) || is_other(f) || is_other(t)
        || (is_directory(f) && is_regular_file(t))
        || detail::stat_equivalent(f_st, t_st))
    {
        return set_or_throw(make_error_code(errc::function_not_supported),
                            ec, "copy", from, to);
    }

    if (ec) ec->clear();

    if (is_symlink(f)) {
        if (bool(copy_options::skip_symlinks & options)) {
            // do nothing
        } else if (not exists(t)) {
            __copy_symlink(from, to, ec);
        } else {
            set_or_throw(make_error_code(errc::file_exists),
                         ec, "copy", from, to);
        }
        return;
    }
    else if (is_regular_file(f)) {
        if (bool(copy_options::directories_only & options)) {
            // do nothing
        }
        else if (bool(copy_options::create_symlinks & options)) {
            __create_symlink(from, to, ec);
        }
        else if (bool(copy_options::create_hard_links & options)) {
            __create_hard_link(from, to, ec);
        }
        else if (is_directory(t)) {
            __copy_file(from, to / from.filename(), options, ec);
        } else {
            __copy_file(from, to, options, ec);
        }
        return;
    }
    else if (is_directory(f) && bool(copy_options::create_symlinks & options)) {
        return set_or_throw(make_error_code(errc::is_a_directory), ec, "copy");
    }
    else if (is_directory(f) && (bool(copy_options::recursive & options) ||
             copy_options::none == options)) {

        if (!exists(t)) {
            // create directory to with attributes from 'from'.
            __create_directory(to, from, ec);
            if (ec && *ec) { return; }
        }
        directory_iterator it = ec ? directory_iterator(from, *ec)
                                   : directory_iterator(from);
        if (ec && *ec) { return; }
        std::error_code m_ec2;
        for (; it != directory_iterator(); it.increment(m_ec2)) {
            if (m_ec2) return set_or_throw(m_ec2, ec, "copy", from, to);
            __copy(it->path(), to / it->path().filename(),
                   options | copy_options::__in_recursive_copy, ec);
            if (ec && *ec) { return; }
        }
    }
}


bool __copy_file(const path& from, const path& to, copy_options options,
                 std::error_code *ec)
{
    using StatT = struct ::stat;
    if (ec)
      ec->clear();

    std::error_code m_ec;
    StatT from_stat;
    auto from_st = detail::posix_stat(from, from_stat, &m_ec);
    if (not is_regular_file(from_st)) {
      if (not m_ec)
        m_ec = make_error_code(errc::not_supported);
      set_or_throw(m_ec, ec, "copy_file", from, to);
      return false;
    }

    StatT to_stat;
    auto to_st = detail::posix_stat(to, to_stat, &m_ec);
    if (!status_known(to_st)) {
        set_or_throw(m_ec, ec, "copy_file", from, to);
        return false;
    }

    const bool to_exists = exists(to_st);
    if (to_exists && !is_regular_file(to_st)) {
        set_or_throw(make_error_code(errc::not_supported), ec, "copy_file", from, to);
        return false;
    }
    if (to_exists && detail::stat_equivalent(from_stat, to_stat)) {
      set_or_throw(make_error_code(errc::file_exists), ec, "copy_file", from,
                   to);
      return false;
    }
    if (to_exists && bool(copy_options::skip_existing & options)) {
        return false;
    }
    else if (to_exists && bool(copy_options::update_existing & options)) {
        auto from_time = __last_write_time(from, ec);
        if (ec && *ec) { return false; }
        auto to_time = __last_write_time(to, ec);
        if (ec && *ec) { return false; }
        if (from_time <= to_time) {
            return false;
        }
        return detail::copy_file_impl(from, to, from_st.permissions(), ec);
    }
    else if (!to_exists || bool(copy_options::overwrite_existing & options)) {
        return detail::copy_file_impl(from, to, from_st.permissions(), ec);
    }
    else {
      set_or_throw(make_error_code(errc::file_exists), ec, "copy_file", from,
                   to);
      return false;
    }

    _LIBCPP_UNREACHABLE();
}

void __copy_symlink(const path& existing_symlink, const path& new_symlink,
                    std::error_code *ec)
{
    const path real_path(__read_symlink(existing_symlink, ec));
    if (ec && *ec) { return; }
    // NOTE: proposal says you should detect if you should call
    // create_symlink or create_directory_symlink. I don't think this
    // is needed with POSIX
    __create_symlink(real_path, new_symlink, ec);
}

bool __create_directories(const path& p, std::error_code *ec)
{
    std::error_code m_ec;
    auto const st = detail::posix_stat(p, &m_ec);
    if (!status_known(st)) {
        set_or_throw(m_ec, ec, "create_directories", p);
        return false;
    }
    else if (is_directory(st)) {
        if (ec) ec->clear();
        return false;
    }
    else if (exists(st)) {
        set_or_throw(make_error_code(errc::file_exists),
                     ec, "create_directories", p);
        return false;
    }

    const path parent = p.parent_path();
    if (!parent.empty()) {
        const file_status parent_st = status(parent, m_ec);
        if (not status_known(parent_st)) {
            set_or_throw(m_ec, ec, "create_directories", p);
            return false;
        }
        if (not exists(parent_st)) {
            __create_directories(parent, ec);
            if (ec && *ec) { return false; }
        }
    }
    return __create_directory(p, ec);
}

bool __create_directory(const path& p, std::error_code *ec)
{
    if (ec) ec->clear();
    if (::mkdir(p.c_str(), static_cast<int>(perms::all)) == 0)
        return true;
    if (errno != EEXIST || !is_directory(p))
        set_or_throw(ec, "create_directory", p);
    return false;
}

bool __create_directory(path const & p, path const & attributes,
                        std::error_code *ec)
{
    struct ::stat attr_stat;
    std::error_code mec;
    auto st = detail::posix_stat(attributes, attr_stat, &mec);
    if (!status_known(st)) {
        set_or_throw(mec, ec, "create_directory", p, attributes);
        return false;
    }
    if (ec) ec->clear();
    if (::mkdir(p.c_str(), attr_stat.st_mode) == 0)
        return true;
    if (errno != EEXIST || !is_directory(p))
        set_or_throw(ec, "create_directory", p, attributes);
    return false;
}

void __create_directory_symlink(path const & from, path const & to,
                                std::error_code *ec){
    if (::symlink(from.c_str(), to.c_str()) != 0)
        set_or_throw(ec, "create_directory_symlink", from, to);
    else if (ec)
        ec->clear();
}

void __create_hard_link(const path& from, const path& to, std::error_code *ec){
    if (::link(from.c_str(), to.c_str()) == -1)
        set_or_throw(ec, "create_hard_link", from, to);
    else if (ec)
        ec->clear();
}

void __create_symlink(path const & from, path const & to, std::error_code *ec) {

    if (::symlink(from.c_str(), to.c_str()) == -1)
        set_or_throw(ec, "create_symlink", from, to);
    else if (ec)
        ec->clear();
}

path __current_path(std::error_code *ec) {
    auto size = ::pathconf(".", _PC_PATH_MAX);
    _LIBCPP_ASSERT(size >= 0, "pathconf returned a 0 as max size");

    auto buff = std::unique_ptr<char[]>(new char[size + 1]);
    char* ret;
    if ((ret = ::getcwd(buff.get(), static_cast<size_t>(size))) == nullptr) {
        set_or_throw(ec, "current_path");
        return {};
    }
    if (ec) ec->clear();
    return {buff.get()};
}

void __current_path(const path& p, std::error_code *ec) {
    if (::chdir(p.c_str()) == -1)
        set_or_throw(ec, "current_path", p);
    else if (ec)
        ec->clear();
}

bool __equivalent(const path& p1, const path& p2, std::error_code *ec)
{
    auto make_unsupported_error = [&]() {
      set_or_throw(make_error_code(errc::not_supported), ec,
                     "equivalent", p1, p2);
      return false;
    };
    std::error_code ec1, ec2;
    struct ::stat st1 = {};
    struct ::stat st2 = {};
    auto s1 = detail::posix_stat(p1.native(), st1, &ec1);
    if (!exists(s1))
      return make_unsupported_error();
    auto s2 = detail::posix_stat(p2.native(), st2, &ec2);
    if (!exists(s2))
      return make_unsupported_error();
    if (ec) ec->clear();
    return detail::stat_equivalent(st1, st2);
}


std::uintmax_t __file_size(const path& p, std::error_code *ec)
{
    std::error_code m_ec;
    struct ::stat st;
    file_status fst = detail::posix_stat(p, st, &m_ec);
    if (!exists(fst) || !is_regular_file(fst)) {
      errc error_kind =
          is_directory(fst) ? errc::is_a_directory : errc::not_supported;
      if (!m_ec)
        m_ec = make_error_code(error_kind);
      set_or_throw(m_ec, ec, "file_size", p);
      return static_cast<uintmax_t>(-1);
    }
    // is_regular_file(p) == true
    if (ec) ec->clear();
    return static_cast<std::uintmax_t>(st.st_size);
}

std::uintmax_t __hard_link_count(const path& p, std::error_code *ec)
{
    std::error_code m_ec;
    struct ::stat st;
    detail::posix_stat(p, st, &m_ec);
    if (m_ec) {
        set_or_throw(m_ec, ec, "hard_link_count", p);
        return static_cast<std::uintmax_t>(-1);
    }
    if (ec) ec->clear();
    return static_cast<std::uintmax_t>(st.st_nlink);
}


bool __fs_is_empty(const path& p, std::error_code *ec)
{
    if (ec) ec->clear();
    std::error_code m_ec;
    struct ::stat pst;
    auto st = detail::posix_stat(p, pst, &m_ec);
    if (m_ec) {
        set_or_throw(m_ec, ec, "is_empty", p);
        return false;
    }
    else if (!is_directory(st) && !is_regular_file(st)) {
        m_ec = make_error_code(errc::not_supported);
        set_or_throw(m_ec, ec, "is_empty");
        return false;
    }
    else if (is_directory(st)) {
        auto it = ec ? directory_iterator(p, *ec) : directory_iterator(p);
        if (ec && *ec)
            return false;
        return it == directory_iterator{};
    }
    else if (is_regular_file(st))
        return static_cast<std::uintmax_t>(pst.st_size) == 0;

    _LIBCPP_UNREACHABLE();
}

static file_time_type __extract_last_write_time(path const& p,
                                                const struct ::stat& st,
                                                error_code *ec) {
  using detail::FSTime;
  auto ts = detail::extract_mtime(st);
  if (!FSTime::is_representable(ts)) {
    set_or_throw(make_error_code(errc::value_too_large), ec, "last_write_time",
                 p);
    return file_time_type::min();
  }
  return FSTime::convert_timespec(ts);
}

file_time_type __last_write_time(const path& p, std::error_code *ec)
{
    using namespace ::std::chrono;
    std::error_code m_ec;
    struct ::stat st;
    detail::posix_stat(p, st, &m_ec);
    if (m_ec) {
        set_or_throw(m_ec, ec, "last_write_time", p);
        return file_time_type::min();
    }
    if (ec) ec->clear();
    return __extract_last_write_time(p, st, ec);
}

void __last_write_time(const path& p, file_time_type new_time,
                       std::error_code *ec)
{
    using namespace std::chrono;
    using namespace detail;

    std::error_code m_ec;
    TimeStructArray tbuf;
#if !defined(_LIBCXX_USE_UTIMENSAT)
    // This implementation has a race condition between determining the
    // last access time and attempting to set it to the same value using
    // ::utimes
    struct ::stat st;
    file_status fst = detail::posix_stat(p, st, &m_ec);
    if (m_ec && !status_known(fst)) {
        set_or_throw(m_ec, ec, "last_write_time", p);
        return;
    }
    SetTimeStructTo(tbuf[0], detail::extract_atime(st));
#else
    tbuf[0].tv_sec = 0;
    tbuf[0].tv_nsec = UTIME_OMIT;
#endif
    if (SetTimeStructTo(tbuf[1], new_time)) {
      set_or_throw(make_error_code(errc::invalid_argument), ec,
                   "last_write_time", p);
      return;
    }

    SetFileTimes(p, tbuf, m_ec);
    if (m_ec)
        set_or_throw(m_ec, ec, "last_write_time", p);
    else if (ec)
        ec->clear();
}


void __permissions(const path& p, perms prms, perm_options opts,
                   std::error_code *ec)
{
    auto has_opt = [&](perm_options o) { return bool(o & opts); };
    const bool resolve_symlinks = !has_opt(perm_options::nofollow);
    const bool add_perms = has_opt(perm_options::add);
    const bool remove_perms = has_opt(perm_options::remove);
    _LIBCPP_ASSERT(
       (add_perms + remove_perms + has_opt(perm_options::replace)) == 1,
       "One and only one of the perm_options constants replace, add, or remove "
        "is present in opts");

    bool set_sym_perms = false;
    prms &= perms::mask;
    if (!resolve_symlinks || (add_perms || remove_perms)) {
        std::error_code m_ec;
        file_status st = resolve_symlinks ? detail::posix_stat(p, &m_ec)
                                          : detail::posix_lstat(p, &m_ec);
        set_sym_perms = is_symlink(st);
        if (m_ec) return set_or_throw(m_ec, ec, "permissions", p);
        _LIBCPP_ASSERT(st.permissions() != perms::unknown,
                       "Permissions unexpectedly unknown");
        if (add_perms)
            prms |= st.permissions();
        else if (remove_perms)
           prms = st.permissions() & ~prms;
    }
    const auto real_perms = detail::posix_convert_perms(prms);

# if defined(AT_SYMLINK_NOFOLLOW) && defined(AT_FDCWD)
    const int flags = set_sym_perms ? AT_SYMLINK_NOFOLLOW : 0;
    if (::fchmodat(AT_FDCWD, p.c_str(), real_perms, flags) == -1) {
        return set_or_throw(ec, "permissions", p);
    }
# else
    if (set_sym_perms)
        return set_or_throw(make_error_code(errc::operation_not_supported),
                            ec, "permissions", p);
    if (::chmod(p.c_str(), real_perms) == -1) {
        return set_or_throw(ec, "permissions", p);
    }
# endif
    if (ec) ec->clear();
}


path __read_symlink(const path& p, std::error_code *ec) {
    char buff[PATH_MAX + 1];
    std::error_code m_ec;
    ::ssize_t ret;
    if ((ret = ::readlink(p.c_str(), buff, PATH_MAX)) == -1) {
        set_or_throw(ec, "read_symlink", p);
        return {};
    }
    _LIBCPP_ASSERT(ret <= PATH_MAX, "TODO");
    _LIBCPP_ASSERT(ret > 0, "TODO");
    if (ec) ec->clear();
    buff[ret] = 0;
    return {buff};
}


bool __remove(const path& p, std::error_code *ec) {
    if (ec) ec->clear();

    if (::remove(p.c_str()) == -1) {
        if (errno != ENOENT)
            set_or_throw(ec, "remove", p);
        return false;
    }
    return true;
}

namespace {

std::uintmax_t remove_all_impl(path const & p, std::error_code& ec)
{
    const auto npos = static_cast<std::uintmax_t>(-1);
    const file_status st = __symlink_status(p, &ec);
    if (ec) return npos;
     std::uintmax_t count = 1;
    if (is_directory(st)) {
        for (directory_iterator it(p, ec); !ec && it != directory_iterator();
             it.increment(ec)) {
            auto other_count = remove_all_impl(it->path(), ec);
            if (ec) return npos;
            count += other_count;
        }
        if (ec) return npos;
    }
    if (!__remove(p, &ec)) return npos;
    return count;
}

} // end namespace

std::uintmax_t __remove_all(const path& p, std::error_code *ec) {
    if (ec) ec->clear();

    std::error_code mec;
    auto count = remove_all_impl(p, mec);
    if (mec) {
        if (mec == errc::no_such_file_or_directory) {
            return 0;
        } else {
            set_or_throw(mec, ec, "remove_all", p);
            return static_cast<std::uintmax_t>(-1);
        }
    }
    return count;
}

void __rename(const path& from, const path& to, std::error_code *ec) {
    if (::rename(from.c_str(), to.c_str()) == -1)
        set_or_throw(ec, "rename", from, to);
    else if (ec)
        ec->clear();
}

void __resize_file(const path& p, std::uintmax_t size, std::error_code *ec) {
    if (::truncate(p.c_str(), static_cast<::off_t>(size)) == -1)
        set_or_throw(ec, "resize_file", p);
    else if (ec)
        ec->clear();
}

space_info __space(const path& p, std::error_code *ec) {
    space_info si;
    struct statvfs m_svfs = {};
    if (::statvfs(p.c_str(), &m_svfs) == -1)  {
        set_or_throw(ec, "space", p);
        si.capacity = si.free = si.available =
            static_cast<std::uintmax_t>(-1);
        return si;
    }
    if (ec) ec->clear();
    // Multiply with overflow checking.
    auto do_mult = [&](std::uintmax_t& out, std::uintmax_t other) {
      out = other * m_svfs.f_frsize;
      if (other == 0 || out / other != m_svfs.f_frsize)
          out = static_cast<std::uintmax_t>(-1);
    };
    do_mult(si.capacity, m_svfs.f_blocks);
    do_mult(si.free, m_svfs.f_bfree);
    do_mult(si.available, m_svfs.f_bavail);
    return si;
}

file_status __status(const path& p, std::error_code *ec) {
    return detail::posix_stat(p, ec);
}

file_status __symlink_status(const path& p, std::error_code *ec) {
    return detail::posix_lstat(p, ec);
}

path __temp_directory_path(std::error_code* ec) {
  const char* env_paths[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};
  const char* ret = nullptr;

  for (auto& ep : env_paths)
    if ((ret = std::getenv(ep)))
      break;
  if (ret == nullptr)
    ret = "/tmp";

  path p(ret);
  std::error_code m_ec;
  if (!exists(p, m_ec) || !is_directory(p, m_ec)) {
    if (!m_ec || m_ec == make_error_code(errc::no_such_file_or_directory))
      m_ec = make_error_code(errc::not_a_directory);
    set_or_throw(m_ec, ec, "temp_directory_path");
    return {};
  }

  if (ec)
    ec->clear();
  return p;
}


path __weakly_canonical(const path& p, std::error_code *ec) {
  if (p.empty())
    return __canonical("", ec);

  path result;
  path tmp;
  tmp.__reserve(p.native().size());
  auto PP = PathParser::CreateEnd(p.native());
  --PP;
  std::vector<string_view_t> DNEParts;

  while (PP.State != PathParser::PS_BeforeBegin) {
    tmp.assign(createView(p.native().data(), &PP.RawEntry.back()));
    std::error_code m_ec;
    file_status st = __status(tmp, &m_ec);
    if (!status_known(st)) {
      set_or_throw(m_ec, ec, "weakly_canonical", p);
      return {};
    } else if (exists(st)) {
      result = __canonical(tmp, ec);
      break;
    }
    DNEParts.push_back(*PP);
    --PP;
  }
  if (PP.State == PathParser::PS_BeforeBegin)
    result = __canonical("", ec);
  if (ec) ec->clear();
  if (DNEParts.empty())
    return result;
  for (auto It=DNEParts.rbegin(); It != DNEParts.rend(); ++It)
    result /= *It;
  return result.lexically_normal();
}

///////////////////////////////////////////////////////////////////////////////
//                            path definitions
///////////////////////////////////////////////////////////////////////////////

constexpr path::value_type path::preferred_separator;

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
    auto PP = PathParser::CreateBegin(__pn_);
    if (PP.State == PathParser::PS_InRootName)
      return *PP;
    return {};
}

string_view_t path::__root_directory() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    if (PP.State == PathParser::PS_InRootName)
      ++PP;
    if (PP.State == PathParser::PS_InRootDir)
      return *PP;
    return {};
}

string_view_t path::__root_path_raw() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    if (PP.State == PathParser::PS_InRootName) {
      auto NextCh = PP.peek();
      if (NextCh && *NextCh == '/') {
        ++PP;
        return createView(__pn_.data(), &PP.RawEntry.back());
      }
      return PP.RawEntry;
    }
    if (PP.State == PathParser::PS_InRootDir)
      return *PP;
    return {};
}

static bool ConsumeRootDir(PathParser* PP) {
  while (PP->State <= PathParser::PS_InRootDir)
    ++(*PP);
  return PP->State == PathParser::PS_AtEnd;
}

string_view_t path::__relative_path() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    if (ConsumeRootDir(&PP))
      return {};
    return createView(PP.RawEntry.data(), &__pn_.back());
}

string_view_t path::__parent_path() const
{
    if (empty())
      return {};
    // Determine if we have a root path but not a relative path. In that case
    // return *this.
    {
      auto PP = PathParser::CreateBegin(__pn_);
      if (ConsumeRootDir(&PP))
        return __pn_;
    }
    // Otherwise remove a single element from the end of the path, and return
    // a string representing that path
    {
      auto PP = PathParser::CreateEnd(__pn_);
      --PP;
      if (PP.RawEntry.data() == __pn_.data())
        return {};
      --PP;
      return createView(__pn_.data(), &PP.RawEntry.back());
    }
}

string_view_t path::__filename() const
{
    if (empty()) return {};
    {
      PathParser PP = PathParser::CreateBegin(__pn_);
      if (ConsumeRootDir(&PP))
        return {};
    }
    return *(--PathParser::CreateEnd(__pn_));
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
// path.gen


enum PathPartKind : unsigned char {
  PK_None,
  PK_RootSep,
  PK_Filename,
  PK_Dot,
  PK_DotDot,
  PK_TrailingSep
};

static PathPartKind ClassifyPathPart(string_view_t Part) {
  if (Part.empty())
    return PK_TrailingSep;
  if (Part == ".")
    return PK_Dot;
  if (Part == "..")
    return PK_DotDot;
  if (Part == "/")
    return PK_RootSep;
  return PK_Filename;
}

path path::lexically_normal() const {
  if (__pn_.empty())
    return *this;

  using PartKindPair = std::pair<string_view_t, PathPartKind>;
  std::vector<PartKindPair> Parts;
  // Guess as to how many elements the path has to avoid reallocating.
  Parts.reserve(32);

  // Track the total size of the parts as we collect them. This allows the
  // resulting path to reserve the correct amount of memory.
  size_t NewPathSize = 0;
  auto AddPart = [&](PathPartKind K, string_view_t P) {
    NewPathSize += P.size();
    Parts.emplace_back(P, K);
  };
  auto LastPartKind = [&]() {
    if (Parts.empty())
      return PK_None;
    return Parts.back().second;
  };

  bool MaybeNeedTrailingSep = false;
  // Build a stack containing the remaining elements of the path, popping off
  // elements which occur before a '..' entry.
  for (auto PP = PathParser::CreateBegin(__pn_); PP; ++PP) {
    auto Part = *PP;
    PathPartKind Kind = ClassifyPathPart(Part);
    switch (Kind) {
    case PK_Filename:
    case PK_RootSep: {
      // Add all non-dot and non-dot-dot elements to the stack of elements.
      AddPart(Kind, Part);
      MaybeNeedTrailingSep = false;
      break;
    }
    case PK_DotDot: {
      // Only push a ".." element if there are no elements preceding the "..",
      // or if the preceding element is itself "..".
      auto LastKind = LastPartKind();
      if (LastKind == PK_Filename) {
        NewPathSize -= Parts.back().first.size();
        Parts.pop_back();
      } else if (LastKind != PK_RootSep)
        AddPart(PK_DotDot, "..");
      MaybeNeedTrailingSep = LastKind == PK_Filename;
      break;
    }
    case PK_Dot:
    case PK_TrailingSep: {
      MaybeNeedTrailingSep = true;
      break;
    }
    case PK_None:
      _LIBCPP_UNREACHABLE();
    }
  }
  // [fs.path.generic]p6.8: If the path is empty, add a dot.
  if (Parts.empty())
     return ".";

  // [fs.path.generic]p6.7: If the last filename is dot-dot, remove any
  // trailing directory-separator.
  bool NeedTrailingSep = MaybeNeedTrailingSep && LastPartKind() == PK_Filename;

  path Result;
  Result.__pn_.reserve(Parts.size() + NewPathSize + NeedTrailingSep);
  for (auto &PK : Parts)
    Result /= PK.first;

  if (NeedTrailingSep)
    Result /= "";

  return Result;
}

static int DetermineLexicalElementCount(PathParser PP) {
  int Count = 0;
  for (; PP; ++PP) {
    auto Elem = *PP;
    if (Elem == "..")
      --Count;
    else if (Elem != ".")
      ++Count;
  }
  return Count;
}

path path::lexically_relative(const path& base) const {
  { // perform root-name/root-directory mismatch checks
    auto PP = PathParser::CreateBegin(__pn_);
    auto PPBase = PathParser::CreateBegin(base.__pn_);
    auto CheckIterMismatchAtBase = [&]() {
        return PP.State != PPBase.State && (
            PP.inRootPath() || PPBase.inRootPath());
    };
    if (PP.State == PathParser::PS_InRootName &&
        PPBase.State == PathParser::PS_InRootName) {
      if (*PP != *PPBase)
        return {};
    } else if (CheckIterMismatchAtBase())
      return {};

    if (PP.inRootPath()) ++PP;
    if (PPBase.inRootPath()) ++PPBase;
    if (CheckIterMismatchAtBase())
      return {};
  }

  // Find the first mismatching element
  auto PP = PathParser::CreateBegin(__pn_);
  auto PPBase = PathParser::CreateBegin(base.__pn_);
  while (PP && PPBase && PP.State == PPBase.State &&
         *PP == *PPBase) {
    ++PP;
    ++PPBase;
  }

  // If there is no mismatch, return ".".
  if (!PP && !PPBase)
    return ".";

  // Otherwise, determine the number of elements, 'n', which are not dot or
  // dot-dot minus the number of dot-dot elements.
  int ElemCount = DetermineLexicalElementCount(PPBase);
  if (ElemCount < 0)
    return {};

  // return a path constructed with 'n' dot-dot elements, followed by the the
  // elements of '*this' after the mismatch.
  path Result;
  // FIXME: Reserve enough room in Result that it won't have to re-allocate.
  while (ElemCount--)
    Result /= "..";
  for (; PP; ++PP)
    Result /= *PP;
  return Result;
}

////////////////////////////////////////////////////////////////////////////
// path.comparisons
int path::__compare(string_view_t __s) const {
    auto PP = PathParser::CreateBegin(__pn_);
    auto PP2 = PathParser::CreateBegin(__s);
    while (PP && PP2) {
        int res = (*PP).compare(*PP2);
        if (res != 0) return res;
        ++PP; ++PP2;
    }
    if (PP.State == PP2.State && !PP)
        return 0;
    if (!PP)
        return -1;
    return 1;
}

////////////////////////////////////////////////////////////////////////////
// path.nonmembers
size_t hash_value(const path& __p) noexcept {
  auto PP = PathParser::CreateBegin(__p.native());
  size_t hash_value = 0;
  std::hash<string_view_t> hasher;
  while (PP) {
    hash_value = __hash_combine(hash_value, hasher(*PP));
    ++PP;
  }
  return hash_value;
}

////////////////////////////////////////////////////////////////////////////
// path.itr
path::iterator path::begin() const
{
    auto PP = PathParser::CreateBegin(__pn_);
    iterator it;
    it.__path_ptr_ = this;
    it.__state_ = PP.State;
    it.__entry_ = PP.RawEntry;
    it.__stashed_elem_.__assign_view(*PP);
    return it;
}

path::iterator path::end() const
{
    iterator it{};
    it.__state_ = PathParser::PS_AtEnd;
    it.__path_ptr_ = this;
    return it;
}

path::iterator& path::iterator::__increment() {
  static_assert(__at_end == PathParser::PS_AtEnd, "");
  PathParser PP(__path_ptr_->native(), __entry_, __state_);
  ++PP;
  __state_ = PP.State;
  __entry_ = PP.RawEntry;
  __stashed_elem_.__assign_view(*PP);
  return *this;
}

path::iterator& path::iterator::__decrement() {
  PathParser PP(__path_ptr_->native(), __entry_, __state_);
  --PP;
  __state_ = PP.State;
  __entry_ = PP.RawEntry;
  __stashed_elem_.__assign_view(*PP);
  return *this;
}

///////////////////////////////////////////////////////////////////////////////
//                           directory entry definitions
///////////////////////////////////////////////////////////////////////////////

#ifndef _LIBCPP_WIN32API
error_code directory_entry::__do_refresh() noexcept {
  __data_.__reset();
  error_code failure_ec;

  struct ::stat full_st;
  file_status st = detail::posix_lstat(__p_, full_st, &failure_ec);
  if (!status_known(st)) {
    __data_.__reset();
    return failure_ec;
  }

  if (!_VSTD_FS::exists(st) || !_VSTD_FS::is_symlink(st)) {
    __data_.__cache_type_ = directory_entry::_RefreshNonSymlink;
    __data_.__type_ = st.type();
    __data_.__non_sym_perms_ = st.permissions();
  } else { // we have a symlink
    __data_.__sym_perms_ = st.permissions();
    // Get the information about the linked entity.
    // Ignore errors from stat, since we don't want errors regarding symlink
    // resolution to be reported to the user.
    error_code ignored_ec;
    st = detail::posix_stat(__p_, full_st, &ignored_ec);

    __data_.__type_ = st.type();
    __data_.__non_sym_perms_ = st.permissions();

    // If we failed to resolve the link, then only partially populate the
    // cache.
    if (!status_known(st)) {
      __data_.__cache_type_ = directory_entry::_RefreshSymlinkUnresolved;
      return error_code{};
    }
    // Otherwise, we resolved the link as not existing. That's OK.
    __data_.__cache_type_ = directory_entry::_RefreshSymlink;
  }

  if (_VSTD_FS::is_regular_file(st))
    __data_.__size_ = static_cast<uintmax_t>(full_st.st_size);

  if (_VSTD_FS::exists(st)) {
    __data_.__nlink_ = static_cast<uintmax_t>(full_st.st_nlink);

    // Attempt to extract the mtime, and fail if it's not representable using
    // file_time_type. For now we ignore the error, as we'll report it when
    // the value is actually used.
    error_code ignored_ec;
    __data_.__write_time_ =
        __extract_last_write_time(__p_, full_st, &ignored_ec);
  }

  return failure_ec;
}
#else
error_code directory_entry::__do_refresh() noexcept {
  __data_.__reset();
  error_code failure_ec;

  file_status st = _VSTD_FS::symlink_status(__p_, failure_ec);
  if (!status_known(st)) {
    __data_.__reset();
    return failure_ec;
  }

  if (!_VSTD_FS::exists(st) || !_VSTD_FS::is_symlink(st)) {
    __data_.__cache_type_ = directory_entry::_RefreshNonSymlink;
    __data_.__type_ = st.type();
    __data_.__non_sym_perms_ = st.permissions();
  } else { // we have a symlink
    __data_.__sym_perms_ = st.permissions();
    // Get the information about the linked entity.
    // Ignore errors from stat, since we don't want errors regarding symlink
    // resolution to be reported to the user.
    error_code ignored_ec;
    st = _VSTD_FS::status(__p_, ignored_ec);

    __data_.__type_ = st.type();
    __data_.__non_sym_perms_ = st.permissions();

    // If we failed to resolve the link, then only partially populate the
    // cache.
    if (!status_known(st)) {
      __data_.__cache_type_ = directory_entry::_RefreshSymlinkUnresolved;
      return error_code{};
    }
    // Otherwise, we resolved the link as not existing. That's OK.
    __data_.__cache_type_ = directory_entry::_RefreshSymlink;
  }

  // FIXME: This is currently broken, and the implementation only a placeholder.
  // We need to cache last_write_time, file_size, and hard_link_count here before
  // the implementation actually works.

  return failure_ec;
}
#endif

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_FILESYSTEM
