
#ifndef MANAGER_HPP
#define MANAGER_HPP

#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/expected.hpp"

#include <string>
#include <vector>

#define BUFFERSIZE 4096

#ifdef _WIN32
// The following variable is used in CreateNamedFunction.
#define PIPE_TIMEOUT 5000
#endif

namespace tl
{
template <typename T, typename U> class expected;
}

namespace N2978
{

enum class ErrorCategory : uint8_t
{
    NONE,

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

struct ProcessMappingOfBMIFile
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
#ifdef _WIN32
    void *hPipe = nullptr;
#else
    int fdSocket = 0;
#endif

    tl::expected<uint32_t, std::string> readInternal(char (&buffer)[BUFFERSIZE]) const;
    tl::expected<void, std::string> writeInternal(const std::vector<char> &buffer) const;

    static std::vector<char> getBufferWithType(CTB type);
    static void writeUInt32(std::vector<char> &buffer, uint32_t value);
    static void writeString(std::vector<char> &buffer, const std::string &str);
    static void writeProcessMappingOfBMIFile(std::vector<char> &buffer, const BMIFile &file);
    static void writeModuleDep(std::vector<char> &buffer, const ModuleDep &dep);
    static void writeHuDep(std::vector<char> &buffer, const HuDep &dep);
    static void writeHeaderFile(std::vector<char> &buffer, const HeaderFile &dep);
    static void writeVectorOfStrings(std::vector<char> &buffer, const std::vector<std::string> &strs);
    static void writeVectorOfProcessMappingOfBMIFiles(std::vector<char> &buffer, const std::vector<BMIFile> &files);
    static void writeVectorOfModuleDep(std::vector<char> &buffer, const std::vector<ModuleDep> &deps);
    static void writeVectorOfHuDeps(std::vector<char> &buffer, const std::vector<HuDep> &deps);
    static void writeVectorOfHeaderFiles(std::vector<char> &buffer, const std::vector<HeaderFile> &headerFiles);

    tl::expected<bool, std::string> readBoolFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                     uint32_t &bytesProcessed) const;
    tl::expected<uint32_t, std::string> readUInt32FromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                           uint32_t &bytesProcessed) const;
    tl::expected<std::string, std::string> readStringFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                              uint32_t &bytesProcessed) const;
    tl::expected<BMIFile, std::string> readProcessMappingOfBMIFileFromPipe(char (&buffer)[BUFFERSIZE],
                                                                           uint32_t &bytesRead,
                                                                           uint32_t &bytesProcessed) const;
    tl::expected<std::vector<std::string>, std::string> readVectorOfStringFromPipe(char (&buffer)[BUFFERSIZE],
                                                                                   uint32_t &bytesRead,
                                                                                   uint32_t &bytesProcessed) const;
    tl::expected<ModuleDep, std::string> readModuleDepFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                               uint32_t &bytesProcessed) const;
    tl::expected<std::vector<ModuleDep>, std::string> readVectorOfModuleDepFromPipe(char (&buffer)[BUFFERSIZE],
                                                                                    uint32_t &bytesRead,
                                                                                    uint32_t &bytesProcessed) const;
    tl::expected<HuDep, std::string> readHuDepFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                       uint32_t &bytesProcessed) const;
    tl::expected<std::vector<HuDep>, std::string> readVectorOfHuDepFromPipe(char (&buffer)[BUFFERSIZE],
                                                                            uint32_t &bytesRead,
                                                                            uint32_t &bytesProcessed) const;
    tl::expected<HeaderFile, std::string> readHeaderFileFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                                 uint32_t &bytesProcessed) const;
    tl::expected<std::vector<HeaderFile>, std::string> readVectorOfHeaderFileFromPipe(char (&buffer)[BUFFERSIZE],
                                                                                      uint32_t &bytesRead,
                                                                                      uint32_t &bytesProcessed) const;
    tl::expected<void, std::string> readNumberOfBytes(char *output, uint32_t size, char (&buffer)[BUFFERSIZE],
                                                      uint32_t &bytesRead, uint32_t &bytesProcessed) const;
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

} // namespace N2978
#endif // MANAGER_HPP
