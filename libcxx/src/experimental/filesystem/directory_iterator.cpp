#include "experimental/filesystem"
#include <dirent.h>
#include <errno.h>

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_FILESYSTEM

namespace { namespace detail {

inline error_code capture_errno() {
    _LIBCPP_ASSERT(errno, "Expected errno to be non-zero");
    return error_code{errno, std::generic_category()};
}

template <class ...Args>
inline bool set_or_throw(std::error_code& my_ec,
                               std::error_code* user_ec,
                               const char* msg, Args&&... args)
{
    if (user_ec) {
        *user_ec = my_ec;
        return true;
    }
    __throw_filesystem_error(msg, std::forward<Args>(args)..., my_ec);
    return false;
}

inline path::string_type posix_readdir(DIR *dir_stream, error_code& ec) {
    struct dirent* dir_entry_ptr = nullptr;
    errno = 0; // zero errno in order to detect errors
    if ((dir_entry_ptr = ::readdir(dir_stream)) == nullptr) {
        ec = capture_errno();
        return {};
    } else {
        ec.clear();
        return dir_entry_ptr->d_name;
    }
}

}}                                                       // namespace detail

using detail::set_or_throw;

class __dir_stream {
public:
    __dir_stream() = delete;
    __dir_stream& operator=(const __dir_stream&) = delete;

    __dir_stream(__dir_stream&& other) noexcept
        : __stream_(other.__stream_), __root_(std::move(other.__root_)),
          __entry_(std::move(other.__entry_))
    {
        other.__stream_ = nullptr;
    }


    __dir_stream(const path& root, directory_options opts, error_code& ec)
        : __stream_(nullptr),
          __root_(root)
    {
        if ((__stream_ = ::opendir(root.c_str())) == nullptr) {
            ec = detail::capture_errno();
            const bool allow_eacess =
                bool(opts & directory_options::skip_permission_denied);
            if (allow_eacess && ec.value() == EACCES)
                ec.clear();
            return;
        }
        advance(ec);
    }

    ~__dir_stream() noexcept
      { if (__stream_) close(); }

    bool good() const noexcept { return __stream_ != nullptr; }

    bool advance(error_code &ec) {
        while (true) {
            auto str = detail::posix_readdir(__stream_,  ec);
            if (str == "." || str == "..") {
                continue;
            } else if (ec || str.empty()) {
                close();
                return false;
            } else {
                __entry_.assign(__root_ / str);
                return true;
            }
        }
    }
private:
    std::error_code close() noexcept {
        std::error_code m_ec;
        if (::closedir(__stream_) == -1)
           m_ec = detail::capture_errno();
        __stream_ = nullptr;
        return m_ec;
    }

    DIR * __stream_{nullptr};
public:
    path __root_;
    directory_entry __entry_;
};

// directory_iterator

directory_iterator::directory_iterator(const path& p, error_code *ec,
                                       directory_options opts)
{
    std::error_code m_ec;
    __imp_ = make_shared<__dir_stream>(p, opts, m_ec);
    if (ec) *ec = m_ec;
    if (!__imp_->good()) {
        __imp_.reset();
        if (m_ec)
            set_or_throw(m_ec, ec,
                         "directory_iterator::directory_iterator(...)", p);
    }
}

directory_iterator& directory_iterator::__increment(error_code *ec)
{
    _LIBCPP_ASSERT(__imp_, "Attempting to increment an invalid iterator");
    std::error_code m_ec;
    if (!__imp_->advance(m_ec)) {
        __imp_.reset();
        if (m_ec)
            set_or_throw(m_ec, ec, "directory_iterator::operator++()");
    } else {
        if (ec) ec->clear();
    }
    return *this;

}

directory_entry const& directory_iterator::__dereference() const {
    _LIBCPP_ASSERT(__imp_, "Attempting to dereference an invalid iterator");
    return __imp_->__entry_;
}

// recursive_directory_iterator

struct recursive_directory_iterator::__shared_imp {
  stack<__dir_stream> __stack_;
  directory_options   __options_;
};

recursive_directory_iterator::recursive_directory_iterator(const path& p,
    directory_options opt, error_code *ec)
    : __imp_(nullptr), __rec_(true)
{
    if (ec) ec->clear();
    std::error_code m_ec;
    __dir_stream new_s(p, opt, m_ec);
    if (m_ec) set_or_throw(m_ec, ec, "recursive_directory_iterator", p);
    if (m_ec || !new_s.good()) return;

    __imp_ = _VSTD::make_shared<__shared_imp>();
    __imp_->__options_ = opt;
    __imp_->__stack_.push(_VSTD::move(new_s));
}

void recursive_directory_iterator::__pop(error_code* ec)
{
    _LIBCPP_ASSERT(__imp_, "Popping the end iterator");
    if (ec) ec->clear();
    __imp_->__stack_.pop();
    if (__imp_->__stack_.size() == 0)
        __imp_.reset();
    else
        __advance(ec);
}

directory_options recursive_directory_iterator::options() const {
    return __imp_->__options_;
}

int recursive_directory_iterator::depth() const {
    return __imp_->__stack_.size() - 1;
}

const directory_entry& recursive_directory_iterator::__dereference() const {
    return __imp_->__stack_.top().__entry_;
}

recursive_directory_iterator&
recursive_directory_iterator::__increment(error_code *ec)
{
    if (ec) ec->clear();
    if (recursion_pending()) {
        if (__try_recursion(ec) || (ec && *ec))
            return *this;
    }
    __rec_ = true;
    __advance(ec);
    return *this;
}

void recursive_directory_iterator::__advance(error_code* ec) {
    // REQUIRES: ec must be cleared before calling this function.
    const directory_iterator end_it;
    auto& stack = __imp_->__stack_;
    std::error_code m_ec;
    while (stack.size() > 0) {
        if (stack.top().advance(m_ec))
            return;
        if (m_ec) break;
        stack.pop();
    }
    __imp_.reset();
    if (m_ec)
        set_or_throw(m_ec, ec, "recursive_directory_iterator::operator++()");
}

bool recursive_directory_iterator::__try_recursion(error_code *ec) {

    bool rec_sym =
        bool(options() & directory_options::follow_directory_symlink);
    auto& curr_it = __imp_->__stack_.top();

    if (is_directory(curr_it.__entry_.status()) &&
        (!is_symlink(curr_it.__entry_.symlink_status()) || rec_sym))
    {
        std::error_code m_ec;
        __dir_stream new_it(curr_it.__entry_.path(), __imp_->__options_, m_ec);
        if (new_it.good()) {
            __imp_->__stack_.push(_VSTD::move(new_it));
            return true;
        }
        if (m_ec) {
            __imp_.reset();
            set_or_throw(m_ec, ec,
                               "recursive_directory_iterator::operator++()");
        }
    }
    return false;
}


_LIBCPP_END_NAMESPACE_EXPERIMENTAL_FILESYSTEM
