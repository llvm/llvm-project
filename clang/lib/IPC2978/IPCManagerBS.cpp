#include "clang/IPC2978/IPCManagerBS.hpp"
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/expected.hpp"
#include <string>
#include <sys/stat.h>

#ifdef _WIN32
#include "clang/IPC2978/rapidhash.h"
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#define TRY_READ(var, func, ...)                                                                                       \
    const auto &var = func(__VA_ARGS__);                                                                               \
    if (!var)                                                                                                          \
    {                                                                                                                  \
        return tl::unexpected(var.error());                                                                            \
    }

#define TRY_READ_VAL(var, func, ...)                                                                                   \
    const auto &var##_result = func(__VA_ARGS__);                                                                      \
    if (!var##_result)                                                                                                 \
    {                                                                                                                  \
        return tl::unexpected(var##_result.error());                                                                   \
    }                                                                                                                  \
    auto &var = *var##_result;

namespace P2978
{

tl::expected<void, std::string> IPCManagerBS::writeInternal(const std::string_view buffer) const
{
#ifdef _WIN32
    const bool success = WriteFile(reinterpret_cast<HANDLE>(writeFd), // pipe handle
                                   buffer.data(),                     // message
                                   buffer.size(),                     // message length
                                   nullptr,                           // bytes written
                                   nullptr);                          // not overlapped
    if (!success)
    {
        return tl::unexpected(getErrorString());
    }
#else
    if (const auto &r = writeAll(writeFd, buffer.data(), buffer.size()); !r)
    {
        return tl::unexpected(r.error());
    }
#endif
    return {};
}

IPCManagerBS::IPCManagerBS(const uint64_t writeFd_) : writeFd(writeFd_)
{
}

tl::expected<void, std::string> IPCManagerBS::receiveMessage(char (&ctbBuffer)[320], CTB &messageType,
                                                             const std::string_view serverReadString)
{
    if (serverReadString.empty())
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }

    uint32_t bytesRead = 1;

    // read call fails if zero byte is read, so safe to process 1 byte
    switch (static_cast<CTB>(serverReadString[0]))
    {

    case CTB::MODULE: {
        TRY_READ_VAL(r, readString, serverReadString, bytesRead);

        messageType = CTB::MODULE;
        getInitializedObjectFromBuffer<CTBModule>(ctbBuffer).moduleName = r;
    }
    break;

    case CTB::NON_MODULE: {
        TRY_READ_VAL(r, readBool, serverReadString, bytesRead);
        TRY_READ_VAL(r2, readString, serverReadString, bytesRead);
        messageType = CTB::NON_MODULE;
        auto &[isHeaderUnit, str] = getInitializedObjectFromBuffer<CTBNonModule>(ctbBuffer);
        isHeaderUnit = r;
        str = r2;
    }
    break;

    case CTB::LAST_MESSAGE: {
        TRY_READ_VAL(fileSizeExpected, readUInt32, serverReadString, bytesRead);

        messageType = CTB::LAST_MESSAGE;
        auto &[fileSize] = getInitializedObjectFromBuffer<CTBLastMessage>(ctbBuffer);
        fileSize = fileSizeExpected;
    }
    break;
    }

    if (serverReadString.size() != bytesRead)
    {
        return tl::unexpected(getErrorString(serverReadString.size(), bytesRead));
    }

    return {};
}

tl::expected<void, std::string> IPCManagerBS::sendMessage(const BTCModule &moduleFile) const
{
    std::string buffer;
    writeBMIFile(buffer, moduleFile.requested);
    buffer.push_back(moduleFile.isSystem);
    writeVectorOfModuleDep(buffer, moduleFile.modDeps);
    buffer.append(delimiter, strlen(delimiter));
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<void, std::string> IPCManagerBS::sendMessage(const BTCNonModule &nonModule) const
{
    std::string buffer;
    buffer.push_back(nonModule.isHeaderUnit);
    buffer.push_back(nonModule.isSystem);
    writeVectorOfHeaderFiles(buffer, nonModule.headerFiles);
    writePath(buffer, nonModule.filePath);
    if (nonModule.isHeaderUnit)
    {
        writeUInt32(buffer, nonModule.fileSize);
        writeVectorOfStrings(buffer, nonModule.logicalNames);
        writeVectorOfHuDeps(buffer, nonModule.huDeps);
    }
    buffer.append(delimiter, strlen(delimiter));
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<void, std::string> IPCManagerBS::sendMessage(const BTCLastMessage &) const
{
    std::string buffer;
    buffer.push_back(true);
    buffer.append(delimiter, strlen(delimiter));
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<Mapping, std::string> IPCManagerBS::createSharedMemoryBMIFile(BMIFile &bmiFile)
{
    Mapping sharedFile{};
#ifdef _WIN32

    // mappingName is needed as the Windows kernel object names can't have \\ in them.
    const uint64_t hash = rapidhash(bmiFile.filePath.data(), bmiFile.filePath.size());
    char mappingName[17];
    static constexpr char hex[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++)
    {
        const uint8_t byte = hash >> (56 - i * 8) & 0xFF;
        mappingName[i * 2] = hex[byte >> 4];
        mappingName[i * 2 + 1] = hex[byte & 0xF];
    }
    mappingName[16] = '\0';

    if (bmiFile.fileSize == UINT32_MAX)
    {
        const HANDLE hFile = CreateFileA(bmiFile.filePath.data(), GENERIC_READ,
                                         0, // no sharing during setup
                                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE)
        {
            return tl::unexpected(getErrorString());
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize))
        {
            return tl::unexpected(getErrorString());
        }

        sharedFile.mapping =
            CreateFileMappingA(hFile, nullptr, PAGE_READONLY, fileSize.HighPart, fileSize.LowPart, mappingName);

        CloseHandle(hFile);

        if (!sharedFile.mapping)
        {
            return tl::unexpected(getErrorString());
        }

        bmiFile.fileSize = fileSize.QuadPart;
        return sharedFile;
    }

    // 1) Open the existing file‐mapping object (must have been created by another process)
    sharedFile.mapping = OpenFileMappingA(FILE_MAP_READ, // read‐only access
                                          FALSE,         // do not inherit handle
                                          mappingName    // name of mapping
    );

    if (sharedFile.mapping == nullptr)
    {
        return tl::unexpected(getErrorString());
    }

    return sharedFile;
#else
    const int fd = open(bmiFile.filePath.data(), O_RDONLY);
    if (fd == -1)
    {
        return tl::unexpected(getErrorString());
    }
    if (bmiFile.fileSize == UINT32_MAX)
    {
        struct stat st;
        if (fstat(fd, &st) == -1)
        {
            return tl::unexpected(getErrorString());
        }

        bmiFile.fileSize = st.st_size;
    }
    void *mapping = mmap(nullptr, bmiFile.fileSize, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (close(fd) == -1)
    {
        return tl::unexpected(getErrorString());
    }
    if (mapping == MAP_FAILED)
    {
        return tl::unexpected(getErrorString());
    }
    sharedFile.file = std::string_view(static_cast<char *>(mapping), bmiFile.fileSize);
    return sharedFile;
#endif
}

tl::expected<void, std::string> IPCManagerBS::closeBMIFileMapping(const Mapping &processMappingOfBMIFile)
{
#ifdef _WIN32
    if (!CloseHandle(processMappingOfBMIFile.mapping))
    {
        return tl::unexpected(getErrorString());
    }
#else
    if (munmap((void *)processMappingOfBMIFile.file.data(), processMappingOfBMIFile.file.size()) == -1)
    {
        return tl::unexpected(getErrorString());
    }
#endif
    return {};
}

} // namespace P2978