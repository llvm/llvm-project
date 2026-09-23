#ifndef IPC2978_RESULT_HPP
#define IPC2978_RESULT_HPP

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace P2978
{

// Explicitly distinguish errors from successful string values, including empty strings.
struct Error
{
    std::string message;
};

// C++17 result with an owned error message. Check success before dereferencing,
// and failure before calling error(); invalid access is a programming error.
// The variant manages value lifetimes and copy/move support, including move-only T.
template <typename T> class [[nodiscard]] Result
{
    std::variant<T, Error> storage;

  public:
    template <typename U = T, std::enable_if_t<std::is_copy_constructible_v<U>, int> = 0>
    Result(const T &value) : storage(std::in_place_index<0>, value)
    {
    }
    template <typename U = T, std::enable_if_t<std::is_move_constructible_v<U>, int> = 0>
    Result(T &&value) : storage(std::in_place_index<0>, std::move(value))
    {
    }
    Result(Error error) : storage(std::in_place_index<1>, std::move(error))
    {
    }

    explicit operator bool() const noexcept
    {
        return storage.index() == 0;
    }

    T *operator->() noexcept
    {
        auto *value = std::get_if<0>(&storage);
        assert(value);
        return value;
    }
    const T *operator->() const noexcept
    {
        const auto *value = std::get_if<0>(&storage);
        assert(value);
        return value;
    }
    T &operator*() & noexcept
    {
        return *operator->();
    }
    const T &operator*() const & noexcept
    {
        return *operator->();
    }
    T &&operator*() && noexcept
    {
        return std::move(*operator->());
    }

    std::string &error() & noexcept
    {
        auto *error = std::get_if<1>(&storage);
        assert(error);
        return error->message;
    }
    const std::string &error() const & noexcept
    {
        const auto *error = std::get_if<1>(&storage);
        assert(error);
        return error->message;
    }
    std::string &&error() && noexcept
    {
        return std::move(this->error());
    }
};

// A void result succeeds with {}; even an empty Error represents failure.
template <> class [[nodiscard]] Result<void>
{
    std::variant<std::monostate, Error> storage;

  public:
    Result() = default;
    Result(Error error) : storage(std::in_place_index<1>, std::move(error))
    {
    }

    explicit operator bool() const noexcept
    {
        return storage.index() == 0;
    }

    std::string &error() & noexcept
    {
        auto *error = std::get_if<1>(&storage);
        assert(error);
        return error->message;
    }
    const std::string &error() const & noexcept
    {
        const auto *error = std::get_if<1>(&storage);
        assert(error);
        return error->message;
    }
    std::string &&error() && noexcept
    {
        return std::move(this->error());
    }
};

} // namespace P2978
#endif // IPC2978_RESULT_HPP
