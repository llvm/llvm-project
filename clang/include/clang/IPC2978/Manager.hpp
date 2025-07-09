
#ifndef MANAGER_HPP
#define MANAGER_HPP

#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/expected.hpp"

#include <string>
#include <vector>

using std::string, std::vector, std::string_view;

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

string getErrorString();
string getErrorString(uint32_t bytesRead_, uint32_t bytesProcessed_);
string getErrorString(ErrorCategory errorCategory_);
// to facilitate error propagation.
inline string getErrorString(string err)
{
    return err;
}

struct ProcessMappingOfBMIFile
{
    string_view file;
#ifdef _WIN32
    void *mapping;
    void *view;
#else
    void *mapping;
    uint32_t mappingSize;
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

    tl::expected<uint32_t, string> readInternal(char (&buffer)[BUFFERSIZE]) const;
    tl::expected<void, string> writeInternal(const vector<char> &buffer) const;

    static vector<char> getBufferWithType(CTB type);
    static void writeUInt32(vector<char> &buffer, uint32_t value);
    static void writeString(vector<char> &buffer, const string &str);
    static void writeProcessMappingOfBMIFile(vector<char> &buffer, const BMIFile &file);
    static void writeModuleDep(vector<char> &buffer, const ModuleDep &dep);
    static void writeHuDep(vector<char> &buffer, const HuDep &dep);
    static void writeVectorOfStrings(vector<char> &buffer, const vector<string> &strs);
    static void writeVectorOfProcessMappingOfBMIFiles(vector<char> &buffer, const vector<BMIFile> &files);
    static void writeVectorOfModuleDep(vector<char> &buffer, const vector<ModuleDep> &deps);
    static void writeVectorOfHuDep(vector<char> &buffer, const vector<HuDep> &deps);

    tl::expected<bool, string> readBoolFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                uint32_t &bytesProcessed) const;
    tl::expected<uint32_t, string> readUInt32FromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                      uint32_t &bytesProcessed) const;
    tl::expected<string, string> readStringFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                    uint32_t &bytesProcessed) const;
    tl::expected<BMIFile, string> readProcessMappingOfBMIFileFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                                      uint32_t &bytesProcessed) const;
    tl::expected<vector<string>, string> readVectorOfStringFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                                    uint32_t &bytesProcessed) const;
    tl::expected<ModuleDep, string> readModuleDepFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                          uint32_t &bytesProcessed) const;
    tl::expected<vector<ModuleDep>, string> readVectorOfModuleDepFromPipe(char (&buffer)[BUFFERSIZE],
                                                                          uint32_t &bytesRead,
                                                                          uint32_t &bytesProcessed) const;
    tl::expected<HuDep, string> readHuDepFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                  uint32_t &bytesProcessed) const;
    tl::expected<vector<HuDep>, string> readVectorOfHuDepFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                                  uint32_t &bytesProcessed) const;
    tl::expected<void, string> readNumberOfBytes(char *output, uint32_t size, char (&buffer)[BUFFERSIZE],
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
