
#ifndef MANAGER_HPP
#define MANAGER_HPP

#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/expected.hpp"

#include <string>
#include <vector>

namespace tl
{
template <typename T, typename U> class expected;
}

namespace P2978
{

// 32-byte delimiter
inline const char *delimiter = "DELIMITER"
                               "\x5A\xA5\x5A\xA5\x5A\xA5\x5A\xA5\x5A\xA5\x5A\xA5\x5A\xA5"
                               "DELIMITER";

enum class ErrorCategory : uint8_t
{
    NONE,

    PARSING_ERROR,
    // error-category for API errors
    READ_FILE_ZERO_BYTES_READ,
    INCORRECT_BTC_LAST_MESSAGE,
    UNKNOWN_CTB_TYPE,
};

std::string getErrorString();
std::string getErrorString(uint32_t bytesRead_, uint32_t bytesProcessed_);
std::string getErrorString(ErrorCategory errorCategory_);
// to facilitate error propagation.
inline std::string getErrorString(std::string err)
{
    return err;
}

struct Mapping
{
    std::string_view file;
#ifdef _WIN32
    void *mapping;
    void *view;
#endif
};

class Manager
{
  public:
    virtual tl::expected<void, std::string> writeInternal(std::string_view buffer) const = 0;
    virtual ~Manager() = default;
#ifndef _WIN32
    static tl::expected<void, std::string> writeAll(const int fd, const char *buffer, const uint32_t count);
#endif

    static std::string getBufferWithType(CTB type);
    static void writeUInt32(std::string &buffer, uint32_t value);
    static void writeString(std::string &buffer, const std::string_view &str);
    // path is used in system calls. so it is followed by null character while the normal string is not.
    static void writePath(std::string &buffer, const std::string_view &str);
    static void writeBMIFile(std::string &buffer, const BMIFile &file);
    static void writeModuleDep(std::string &buffer, const ModuleDep &dep);
    static void writeHuDep(std::string &buffer, const HuDep &dep);
    static void writeHeaderFile(std::string &buffer, const HeaderFile &dep);
    static void writeVectorOfStrings(std::string &buffer, const std::vector<std::string_view> &strs);
    static void writeVectorOfProcessMappingOfBMIFiles(std::string &buffer, const std::vector<BMIFile> &files);
    static void writeVectorOfModuleDep(std::string &buffer, const std::vector<ModuleDep> &deps);
    static void writeVectorOfHuDeps(std::string &buffer, const std::vector<HuDep> &deps);
    static void writeVectorOfHeaderFiles(std::string &buffer, const std::vector<HeaderFile> &headerFiles);

    static tl::expected<bool, std::string> readBool(std::string_view message, uint32_t &bytesRead);
    static tl::expected<uint32_t, std::string> readUInt32(std::string_view message, uint32_t &bytesRead);
    static tl::expected<std::string_view, std::string> readString(std::string_view message, uint32_t &bytesRead);

    // path is used in system calls. so it is followed by null character while the normal string is not.
    static tl::expected<std::string_view, std::string> readPath(std::string_view message, uint32_t &bytesRead);
};

template <typename T, typename... Args> constexpr T *construct_at(T *p, Args &&...args)
{
    return ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
}

template <typename T> T &getInitializedObjectFromBuffer(char (&buffer)[320])
{
    T &t = reinterpret_cast<T &>(buffer);
    construct_at(&t);
    return t;
}

inline std::string to16charHexString(const uint64_t v)
{
    static auto lut = "0123456789abcdef";
    std::string out;
    out.resize(16);
    for (int i = 0; i < 8; ++i)
    {
        // extract byte in big-endian order:
        const auto byte = static_cast<uint8_t>(v >> ((7 - i) * 8));
        // high nibble:
        out[2 * i] = lut[byte >> 4];
        // low nibble:
        out[2 * i + 1] = lut[byte & 0xF];
    }
    return out;
}

} // namespace P2978
#endif // MANAGER_HPP
