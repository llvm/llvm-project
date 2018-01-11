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
#include "type_traits"
#include "random"  /* for unique_path */
#include "cstdlib"
#include "climits"

#include "filesystem_time_helper.h"

#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>  /* values for fchmodat */
#if !defined(UTIME_OMIT)
#include <sys/time.h> // for ::utimes as used in __last_write_time
#endif

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_FILESYSTEM

filesystem_error::~filesystem_error() {}


//                       POSIX HELPERS

namespace detail { namespace  {

using value_type = path::value_type;
using string_type = path::string_type;

inline std::error_code capture_errno() {
  _LIBCPP_ASSERT(errno, "Expected errno to be non-zero");
  return std::error_code(errno, std::generic_category());
}

void set_or_throw(std::error_code const& m_ec, std::error_code* ec,
                  const char* msg, path const& p = {}, path const& p2 = {})
{
    if (ec) {
        *ec = m_ec;
    } else {
        string msg_s("std::experimental::filesystem::");
        msg_s += msg;
        __throw_filesystem_error(msg_s, p, p2, m_ec);
    }
}

void set_or_throw(std::error_code* ec, const char* msg,
                  path const& p = {}, path const& p2 = {})
{
    return set_or_throw(capture_errno(), ec, msg, p, p2);
}

perms posix_get_perms(const struct ::stat & st) noexcept {
    return static_cast<perms>(st.st_mode) & perms::mask;
}

::mode_t posix_convert_perms(perms prms) {
    return static_cast< ::mode_t>(prms & perms::mask);
}

file_status create_file_status(std::error_code& m_ec, path const& p,
                               struct ::stat& path_stat,
                               std::error_code* ec)
{
    if (ec) *ec = m_ec;
    if (m_ec && (m_ec.value() == ENOENT || m_ec.value() == ENOTDIR)) {
        return file_status(file_type::not_found);
    }
    else if (m_ec) {
        set_or_throw(m_ec, ec, "posix_stat", p);
        return file_status(file_type::none);
    }
    // else

    file_status fs_tmp;
    auto const mode = path_stat.st_mode;
    if      (S_ISLNK(mode))  fs_tmp.type(file_type::symlink);
    else if (S_ISREG(mode))  fs_tmp.type(file_type::regular);
    else if (S_ISDIR(mode))  fs_tmp.type(file_type::directory);
    else if (S_ISBLK(mode))  fs_tmp.type(file_type::block);
    else if (S_ISCHR(mode))  fs_tmp.type(file_type::character);
    else if (S_ISFIFO(mode)) fs_tmp.type(file_type::fifo);
    else if (S_ISSOCK(mode)) fs_tmp.type(file_type::socket);
    else                     fs_tmp.type(file_type::unknown);

    fs_tmp.permissions(detail::posix_get_perms(path_stat));
    return fs_tmp;
}

file_status posix_stat(path const & p, struct ::stat& path_stat,
                       std::error_code* ec)
{
    std::error_code m_ec;
    if (::stat(p.c_str(), &path_stat) == -1)
        m_ec = detail::capture_errno();
    return create_file_status(m_ec, p, path_stat, ec);
}

file_status posix_stat(path const & p, std::error_code* ec) {
    struct ::stat path_stat;
    return posix_stat(p, path_stat, ec);
}

file_status posix_lstat(path const & p, struct ::stat & path_stat,
                        std::error_code* ec)
{
    std::error_code m_ec;
    if (::lstat(p.c_str(), &path_stat) == -1)
        m_ec = detail::capture_errno();
    return create_file_status(m_ec, p, path_stat, ec);
}

file_status posix_lstat(path const & p, std::error_code* ec) {
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
    __permissions(to, from_perms, ec);
    // TODO what if permissions fails?
    return true;
}

}} // end namespace detail

using detail::set_or_throw;

path __canonical(path const & orig_p, const path& base, std::error_code *ec)
{
    path p = absolute(orig_p, base);
    char buff[PATH_MAX + 1];
    char *ret;
    if ((ret = ::realpath(p.c_str(), buff)) == nullptr) {
        set_or_throw(ec, "canonical", orig_p, base);
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
    if (ec) ec->clear();

    std::error_code m_ec;
    auto from_st = detail::posix_stat(from, &m_ec);
    if (not is_regular_file(from_st)) {
        if (not m_ec)
            m_ec = make_error_code(errc::not_supported);
        set_or_throw(m_ec, ec, "copy_file", from, to);
        return false;
    }

    auto to_st = detail::posix_stat(to, &m_ec);
    if (!status_known(to_st)) {
        set_or_throw(m_ec, ec, "copy_file", from, to);
        return false;
    }

    const bool to_exists = exists(to_st);
    if (to_exists && !is_regular_file(to_st)) {
        set_or_throw(make_error_code(errc::not_supported), ec, "copy_file", from, to);
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
        set_or_throw(make_error_code(errc::file_exists), ec, "copy", from, to);
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
    return (st1.st_dev == st2.st_dev && st1.st_ino == st2.st_ino);
}


std::uintmax_t __file_size(const path& p, std::error_code *ec)
{
    std::error_code m_ec;
    struct ::stat st;
    file_status fst = detail::posix_stat(p, st, &m_ec);
    if (!exists(fst) || !is_regular_file(fst)) {
        if (!m_ec)
            m_ec = make_error_code(errc::not_supported);
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


namespace detail { namespace {

using TimeSpec = struct timespec;
using StatT =  struct stat;

#if defined(__APPLE__)
TimeSpec extract_mtime(StatT const& st) { return st.st_mtimespec; }
__attribute__((unused)) // Suppress warning
TimeSpec extract_atime(StatT const& st) { return st.st_atimespec; }
#else
TimeSpec extract_mtime(StatT const& st) { return st.st_mtim; }
__attribute__((unused)) // Suppress warning
TimeSpec extract_atime(StatT const& st) { return st.st_atim; }
#endif

}} // end namespace detail

using FSTime = fs_time_util<file_time_type, time_t, struct timespec>;

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
    auto ts = detail::extract_mtime(st);
    if (!FSTime::is_representable(ts)) {
        set_or_throw(error_code(EOVERFLOW, generic_category()), ec,
                     "last_write_time", p);
        return file_time_type::min();
    }
    return FSTime::convert_timespec(ts);
}

void __last_write_time(const path& p, file_time_type new_time,
                       std::error_code *ec)
{
    using namespace std::chrono;
    std::error_code m_ec;

    // We can use the presence of UTIME_OMIT to detect platforms that do not
    // provide utimensat.
#if !defined(UTIME_OMIT)
    // This implementation has a race condition between determining the
    // last access time and attempting to set it to the same value using
    // ::utimes
    struct ::stat st;
    file_status fst = detail::posix_stat(p, st, &m_ec);
    if (m_ec && !status_known(fst)) {
        set_or_throw(m_ec, ec, "last_write_time", p);
        return;
    }
    auto atime = detail::extract_atime(st);
    struct ::timeval tbuf[2];
    tbuf[0].tv_sec = atime.tv_sec;
    tbuf[0].tv_usec = duration_cast<microseconds>(nanoseconds(atime.tv_nsec)).count();
    const bool overflowed = !FSTime::set_times_checked<microseconds>(
        &tbuf[1].tv_sec, &tbuf[1].tv_usec, new_time);

    if (overflowed) {
        set_or_throw(make_error_code(errc::invalid_argument), ec,
                     "last_write_time", p);
        return;
    }
    if (::utimes(p.c_str(), tbuf) == -1) {
        m_ec = detail::capture_errno();
    }
#else
    struct ::timespec tbuf[2];
    tbuf[0].tv_sec = 0;
    tbuf[0].tv_nsec = UTIME_OMIT;

    const bool overflowed = !FSTime::set_times_checked<nanoseconds>(
        &tbuf[1].tv_sec, &tbuf[1].tv_nsec, new_time);
    if (overflowed) {
        set_or_throw(make_error_code(errc::invalid_argument),
            ec, "last_write_time", p);
        return;
    }
    if (::utimensat(AT_FDCWD, p.c_str(), tbuf, 0) == -1) {
        m_ec = detail::capture_errno();
    }
#endif
    if (m_ec)
        set_or_throw(m_ec, ec, "last_write_time", p);
    else if (ec)
        ec->clear();
}


void __permissions(const path& p, perms prms, std::error_code *ec)
{

    const bool resolve_symlinks = !bool(perms::symlink_nofollow & prms);
    const bool add_perms = bool(perms::add_perms & prms);
    const bool remove_perms = bool(perms::remove_perms & prms);
    _LIBCPP_ASSERT(!(add_perms && remove_perms),
                   "Both add_perms and remove_perms are set");

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

path __system_complete(const path& p, std::error_code *ec) {
    if (ec) ec->clear();
    return absolute(p, current_path());
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

// An absolute path is composed according to the table in [fs.op.absolute].
path absolute(const path& p, const path& base) {
    auto root_name = p.root_name();
    auto root_dir = p.root_directory();

    if (!root_name.empty() && !root_dir.empty())
      return p;

    auto abs_base = base.is_absolute() ? base : absolute(base);

    /* !has_root_name && !has_root_dir */
    if (root_name.empty() && root_dir.empty())
    {
      return abs_base / p;
    }
    else if (!root_name.empty()) /* has_root_name && !has_root_dir */
    {
      return  root_name / abs_base.root_directory()
              /
              abs_base.relative_path() / p.relative_path();
    }
    else /* !has_root_name && has_root_dir */
    {
      if (abs_base.has_root_name())
        return abs_base.root_name() / p;
      // else p is absolute,  return outside of block
    }
    return p;
}

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_FILESYSTEM
