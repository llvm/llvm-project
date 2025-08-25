
#ifndef IPC_MANAGER_COMPILER_HPP
#define IPC_MANAGER_COMPILER_HPP

#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/expected.hpp"

using std::string_view;
namespace N2978
{

// IPC Manager Compiler
class IPCManagerCompiler : Manager
{
    template <typename T> tl::expected<T, string> receiveMessage() const;
    // This is not exposed. sendCTBLastMessage calls this.
    [[nodiscard]] tl::expected<void, string> receiveBTCLastMessage() const;

  public:
    CTBLastMessage lastMessage{};
#ifdef _WIN32
    explicit IPCManagerCompiler(void *hPipe_);
#else
    explicit IPCManagerCompiler(int fdSocket_);
#endif
    [[nodiscard]] tl::expected<BTCModule, string> receiveBTCModule(const CTBModule &moduleName) const;
    [[nodiscard]] tl::expected<BTCNonModule, string> receiveBTCNonModule(const CTBNonModule &nonModule) const;
    [[nodiscard]] tl::expected<void, string> sendCTBLastMessage(const CTBLastMessage &lastMessage) const;
    [[nodiscard]] tl::expected<void, string> sendCTBLastMessage(const CTBLastMessage &lastMessage,
                                                                const string &bmiFile, const string &filePath) const;
    static tl::expected<ProcessMappingOfBMIFile, string> readSharedMemoryBMIFile(const BMIFile &file);
    static tl::expected<void, string> closeBMIFileMapping(const ProcessMappingOfBMIFile &processMappingOfBMIFile);
    void closeConnection() const;
};

template <typename T> tl::expected<T, string> IPCManagerCompiler::receiveMessage() const
{
    // Read from the pipe.
    char buffer[BUFFERSIZE];
    uint32_t bytesRead;
    if (const auto &r = readInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    else
    {
        bytesRead = *r;
    }

    uint32_t bytesProcessed = 0;

    if constexpr (std::is_same_v<T, BTCModule>)
    {
        const auto &r = readProcessMappingOfBMIFileFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }

        const auto &r2 = readVectorOfModuleDepFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r2)
        {
            return tl::unexpected(r2.error());
        }

        BTCModule moduleFile;
        moduleFile.requested = *r;
        moduleFile.deps = *r2;
        if (bytesRead == bytesProcessed)
        {
            return moduleFile;
        }
    }
    else if constexpr (std::is_same_v<T, BTCNonModule>)
    {
        const auto &r = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }

        const auto &r2 = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r2)
        {
            return tl::unexpected(r2.error());
        }

        const auto &r3 = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r3)
        {
            return tl::unexpected(r3.error());
        }

        const auto &r4 = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
        if (!r4)
        {
            return tl::unexpected(r4.error());
        }

        const auto &r5 = readVectorOfHuDepFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r5)
        {
            return tl::unexpected(r5.error());
        }

        BTCNonModule nonModule;
        nonModule.isHeaderUnit = *r;
        nonModule.filePath = *r2;
        nonModule.user = *r3;
        nonModule.fileSize = *r4;
        nonModule.deps = *r5;

        if (bytesRead == bytesProcessed)
        {
            return nonModule;
        }
    }
    else
    {
        static_assert(false && "Unknown type\n");
    }

    if (bytesRead != bytesProcessed)
    {
        return tl::unexpected(getErrorString(bytesRead, bytesProcessed));
    }
    string str = __FILE__;
    str += ':';
    str += std::to_string(__LINE__);
    return tl::unexpected(getErrorString("N2978 IPC API internal error" + str));
}
[[nodiscard]] tl::expected<IPCManagerCompiler, string> makeIPCManagerCompiler(string BMIIfHeaderUnitObjOtherwisePath);
inline IPCManagerCompiler *managerCompiler;
inline CTBLastMessage lastMessage;

// Equality operator for use in unordered_map
bool operator==(const CTBNonModule &lhs, const CTBNonModule &rhs);
// Hash function for CTBNonModule
struct CTBNonModuleHash
{
    uint64_t operator()(const CTBNonModule &ctb) const;
};

inline std::unordered_map<CTBNonModule, BTCNonModule, CTBNonModuleHash> respnses;
} // namespace N2978
#endif // IPC_MANAGER_COMPILER_HPP
