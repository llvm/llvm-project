
#ifndef IPC_MANAGER_COMPILER_HPP
#define IPC_MANAGER_COMPILER_HPP

#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/expected.hpp"

struct CompilerTest;
namespace N2978
{

enum class FileType : uint8_t
{
    MODULE,
    HEADER_UNIT,
    HEADER_FILE
};

struct Response
{
    // if type == HEADER_FILE, then fileSize has no meaning
    BMIFile file;
    FileType type;
    bool user;
    Response(BMIFile file_, FileType type_, bool user_);
};

// IPC Manager Compiler
class IPCManagerCompiler : Manager
{
    friend struct ::CompilerTest;
    template <typename T> tl::expected<T, std::string> receiveMessage() const;
    // This is not exposed. sendCTBLastMessage calls this.
    [[nodiscard]] tl::expected<void, std::string> receiveBTCLastMessage() const;
    [[nodiscard]] tl::expected<BTCModule, std::string> receiveBTCModule(const CTBModule &moduleName);
    [[nodiscard]] tl::expected<BTCNonModule, std::string> receiveBTCNonModule(const CTBNonModule &nonModule);

    std::unordered_map<std::string, Response> responses;

  public:
    CTBLastMessage lastMessage{};
#ifdef _WIN32
    explicit IPCManagerCompiler(void *hPipe_);
#else
    explicit IPCManagerCompiler(int fdSocket_);
#endif

    // For FileType::HEADER_FILE, it can return FileType::HEADER_UNIT, otherwise it will return the request
    // response. Either it will return from the cache or it will fetch it from the build-system
    [[nodiscard]] tl::expected<Response, std::string> findResponse(const std::string &logicalName, FileType type);
    [[nodiscard]] tl::expected<void, std::string> sendCTBLastMessage(const CTBLastMessage &lastMessage) const;
    [[nodiscard]] tl::expected<void, std::string> sendCTBLastMessage(const CTBLastMessage &lastMessage,
                                                                     const std::string &bmiFile,
                                                                     const std::string &filePath) const;
    static tl::expected<ProcessMappingOfBMIFile, std::string> readSharedMemoryBMIFile(const BMIFile &file);
    static tl::expected<void, std::string> closeBMIFileMapping(const ProcessMappingOfBMIFile &processMappingOfBMIFile);
    void closeConnection() const;
};

template <typename T> tl::expected<T, std::string> IPCManagerCompiler::receiveMessage() const
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

        const auto &r2 = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r2)
        {
            return tl::unexpected(r2.error());
        }

        const auto &r3 = readVectorOfModuleDepFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r3)
        {
            return tl::unexpected(r3.error());
        }

        BTCModule moduleFile;
        moduleFile.requested = *r;
        moduleFile.user = *r2;
        moduleFile.modDeps = *r3;

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

        const auto &r2 = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r2)
        {
            return tl::unexpected(r2.error());
        }

        const auto &r3 = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r3)
        {
            return tl::unexpected(r3.error());
        }

        const auto &r4 = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
        if (!r4)
        {
            return tl::unexpected(r4.error());
        }

        const auto &r5 = readVectorOfStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r5)
        {
            return tl::unexpected(r5.error());
        }

        const auto &r6 = readVectorOfHeaderFileFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r6)
        {
            return tl::unexpected(r6.error());
        }

        const auto &r7 = readVectorOfHuDepFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r7)
        {
            return tl::unexpected(r7.error());
        }

        BTCNonModule nonModule;
        nonModule.isHeaderUnit = *r;
        nonModule.user = *r2;
        nonModule.filePath = *r3;
        nonModule.fileSize = *r4;
        nonModule.logicalNames = *r5;
        nonModule.headerFiles = *r6;
        nonModule.huDeps = *r7;

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
    std::string str = __FILE__;
    str += ':';
    str += std::to_string(__LINE__);
    return tl::unexpected(getErrorString("N2978 IPC API internal error" + str));
}
[[nodiscard]] tl::expected<IPCManagerCompiler, std::string> makeIPCManagerCompiler(
    std::string BMIIfHeaderUnitObjOtherwisePath);
inline IPCManagerCompiler *managerCompiler;
} // namespace N2978
#endif // IPC_MANAGER_COMPILER_HPP
