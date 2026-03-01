
#include "clang/IPC2978/IPCManagerCompiler.hpp"
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"

#include <string>
#include <utility>

#ifdef _WIN32
#include "clang/IPC2978/rapidhash.h"
#include <Windows.h>
#else
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
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

Response::Response(std::string_view filePath_, const Mapping &mapping_, const FileType type_, const bool isSystem_)
    : filePath(std::move(filePath_)), mapping(mapping_), type(type_), isSystem(isSystem_)
{
}

static bool endsWith(const std::string_view str, const std::string &suffix)
{
    if (suffix.size() > str.size())
    {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

tl::expected<std::string_view, std::string> IPCManagerCompiler::readInternal(char (&buffer)[4096]) const
{
    std::string *output = nullptr;
    while (true)
    {
        uint32_t bytesRead;
#ifdef _WIN32
        const bool success = ReadFile((HANDLE)STD_INPUT_HANDLE, // pipe handle
                                      buffer,                   // buffer to receive reply
                                      4096,                     // size of buffer
                                      LPDWORD(&bytesRead),      // number of bytes read
                                      nullptr);                 // not overlapped

        if (const uint32_t lastError = GetLastError(); !success && lastError != ERROR_MORE_DATA)
        {
            return tl::unexpected(getErrorString());
        }

#else
        bytesRead = read(STDIN_FILENO, buffer, 4096);
        if (bytesRead == -1)
        {
            return tl::unexpected(getErrorString());
        }

#endif
        if (!bytesRead)
        {
            return tl::unexpected(getErrorString(ErrorCategory::READ_FILE_ZERO_BYTES_READ));
        }

        if (!output)
        {
            if (bytesRead < strlen(delimiter))
            {
                return tl::unexpected("P2978 Error: Received string only has delimiter but not the size of payload\n");
            }
            output = new std::string{};
            allocations.emplace_back(output);
        }

        output->append(buffer, bytesRead);

        // We return once we receive the delimiter.
        if (endsWith(*output, delimiter))
        {
            return std::string_view{output->data(), output->size() - strlen(delimiter)};
        }
    }
}

tl::expected<void, std::string> IPCManagerCompiler::writeInternal(const std::string_view buffer) const
{
#ifdef _WIN32
    const bool success = WriteFile(reinterpret_cast<HANDLE>(STD_OUTPUT_HANDLE), // pipe handle
                                   buffer.data(),                               // message
                                   buffer.size(),                               // message length
                                   nullptr,                                     // bytes written
                                   nullptr);                                    // not overlapped
    if (!success)
    {
        return tl::unexpected(getErrorString());
    }
#else
    if (const auto &r = writeAll(STDOUT_FILENO, buffer.data(), buffer.size()); !r)
    {
        return tl::unexpected(r.error());
    }
#endif
    return {};
}

tl::expected<IPCManagerCompiler::BMIFileMapping, std::string> IPCManagerCompiler::readProcessMappingOfBMIFile(
    const std::string_view message, uint32_t &bytesRead)
{
    const auto &r = readPath(message, bytesRead);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    const auto &r2 = readUInt32(message, bytesRead);
    if (!r2)
    {
        return tl::unexpected(r2.error());
    }

    BMIFile file;
    file.filePath = *r;
    file.fileSize = *r2;

    if (const auto &r3 = readSharedMemoryBMIFile(file); r3)
    {
        filePathProcessMapping.emplace(file.filePath, r3.value());
        BMIFileMapping bmiFileMapping;
        bmiFileMapping.file = file;
        bmiFileMapping.mapping = *r3;
        return bmiFileMapping;
    }
    else
    {
        return tl::unexpected(r3.error());
    }
}

tl::expected<void, std::string> IPCManagerCompiler::readLogicalNames(const std::string_view message,
                                                                     uint32_t &bytesRead, const BMIFileMapping &mapping,
                                                                     const FileType type, const bool isSystem)
{
    TRY_READ_VAL(logicalNamesSize, readUInt32, message, bytesRead);
    for (uint32_t i = 0; i < logicalNamesSize; ++i)
    {
        TRY_READ_VAL(logicalName, readString, message, bytesRead);
        responses.emplace(logicalName, Response(mapping.file.filePath, mapping.mapping, type, isSystem));
    }

    return {};
}

tl::expected<void, std::string> IPCManagerCompiler::receiveBTCLastMessage() const
{
    char buffer[4096];
    const auto &r = readInternal(buffer);
    if (!r)
    {
        return tl::unexpected(r.error());
    }

    // The BTCLastMessage must be 1 byte of true signaling that build-system has successfully created a shared memory
    // mapping of the BMI file.
    if (buffer[0] != static_cast<char>(true))
    {
        return tl::unexpected(getErrorString(ErrorCategory::INCORRECT_BTC_LAST_MESSAGE));
    }

    if (r->size() != 1)
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }

    return {};
}

tl::expected<void, std::string> IPCManagerCompiler::receiveBTCModule(const CTBModule &moduleName)
{
    std::string buffer = getBufferWithType(CTB::MODULE);
    writeString(buffer, moduleName.moduleName);
    writeUInt32(buffer, buffer.size());
    buffer.append(delimiter, strlen(delimiter));
    // This call sends the CTBModule to the build-system.
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }

    char stackBuffer[4096];
    auto received = readInternal(stackBuffer);

    if (!received)
    {
        return tl::unexpected(received.error());
    }
    const std::string_view message = *received;

    uint32_t bytesRead = 0;

    TRY_READ_VAL(requested, readProcessMappingOfBMIFile, message, bytesRead);
    TRY_READ_VAL(isSystem, readBool, message, bytesRead);

    std::string *str = new std::string(moduleName.moduleName);
    allocations.emplace_back(str);
    responses.emplace(*str, Response(requested.file.filePath, requested.mapping, FileType::MODULE, isSystem));

    TRY_READ_VAL(modDepsSize, readUInt32, message, bytesRead);

    for (uint32_t i = 0; i < modDepsSize; ++i)
    {
        TRY_READ_VAL(isHeaderUnit, readBool, message, bytesRead);
        TRY_READ_VAL(modDepFile, readProcessMappingOfBMIFile, message, bytesRead);
        TRY_READ_VAL(modDepIsSytem, readBool, message, bytesRead);
        if (isHeaderUnit)
        {
            if (const auto &r = readLogicalNames(message, bytesRead, modDepFile, FileType::HEADER_UNIT, modDepIsSytem);
                !r)
            {
                return tl::unexpected(r.error());
            }
        }
        else
        {
            if (const auto &r = readLogicalNames(message, bytesRead, modDepFile, FileType::MODULE, modDepIsSytem); !r)
            {
                return tl::unexpected(r.error());
            }
        }
    }

    if (message.size() != bytesRead)
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }
    return {};
}

tl::expected<void, std::string> IPCManagerCompiler::receiveBTCNonModule(const CTBNonModule &nonModule)
{
    std::string buffer = getBufferWithType(CTB::NON_MODULE);
    buffer.push_back(nonModule.isHeaderUnit);
    writeString(buffer, nonModule.logicalName);
    writeUInt32(buffer, buffer.size());
    buffer.append(delimiter, strlen(delimiter));
    // This call sends the CTBNonModule to the build-system.
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }

    char stackBuffer[4096];
    auto received = readInternal(stackBuffer);

    if (!received)
    {
        return tl::unexpected(received.error());
    }

    std::string_view readCompilerMessage = *received;
    uint32_t bytesRead = 0;

    TRY_READ_VAL(isHeaderUnit, readBool, readCompilerMessage, bytesRead);
    TRY_READ_VAL(isSystem, readBool, readCompilerMessage, bytesRead);
    TRY_READ_VAL(headerFilesSize, readUInt32, readCompilerMessage, bytesRead);

    for (uint32_t i = 0; i < headerFilesSize; ++i)
    {
        TRY_READ_VAL(logicalName, readString, readCompilerMessage, bytesRead);
        TRY_READ_VAL(filePath, readPath, readCompilerMessage, bytesRead);
        TRY_READ_VAL(isSystemHeaderFile, readBool, readCompilerMessage, bytesRead);

        responses.emplace(logicalName, Response{filePath, {}, FileType::HEADER_FILE, isSystemHeaderFile});
    }

    std::string *str = new std::string(nonModule.logicalName);
    allocations.emplace_back(str);
    if (!isHeaderUnit)
    {
        TRY_READ_VAL(filePath, readPath, readCompilerMessage, bytesRead);
        responses.emplace(*str, Response{filePath, {}, FileType::HEADER_FILE, isSystem});
        if (readCompilerMessage.size() != bytesRead)
        {
            return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
        }
        return {};
    }

    TRY_READ_VAL(file, readProcessMappingOfBMIFile, readCompilerMessage, bytesRead);
    responses.emplace(*str, Response{file.file.filePath, file.mapping, FileType::HEADER_UNIT, isSystem});

    TRY_READ(logicalNames, readLogicalNames, readCompilerMessage, bytesRead, file, FileType::HEADER_UNIT, isSystem);

    TRY_READ_VAL(huDepsSize, readUInt32, readCompilerMessage, bytesRead);
    for (uint32_t i = 0; i < huDepsSize; ++i)
    {
        TRY_READ_VAL(huDepFile, readProcessMappingOfBMIFile, readCompilerMessage, bytesRead);
        TRY_READ_VAL(huDepIsSystem, readBool, readCompilerMessage, bytesRead);
        TRY_READ(huDeplogicalNames, readLogicalNames, readCompilerMessage, bytesRead, huDepFile, FileType::HEADER_UNIT,
                 huDepIsSystem);
    }

    if (readCompilerMessage.size() != bytesRead)
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }
    return {};
}

tl::expected<Response, std::string> IPCManagerCompiler::findResponse(std::string_view logicalName, const FileType type)
{
#ifdef _WIN32
    std::string logicalName2{logicalName};
    if (type != FileType::MODULE)
    {
        for (char &c : logicalName2)
        {
            c = std::tolower(c);
        }
    }
#endif

    if (const auto &it = responses.find(logicalName);
        // This requests from the build-system if we don't have an entry for the logicalName or if there is a type
        // mismatch between the request and the response. Only allowed mismatch is if the request is of header-file and
        // the response is a header-unit instead. For other mismatches compiler will request the build-system which will
        // give not found error. HMake at config-time checks for the logicalName collision and also that a file is not
        // registered as 2 of header-file, header-unit and module.
        it == responses.end() ||
        (it->second.type != type && (it->second.type != FileType::HEADER_UNIT || type != FileType::HEADER_FILE)))
    {
        if (type == FileType::MODULE)
        {
            CTBModule ctbModule;
            ctbModule.moduleName = logicalName;
            if (const auto &r2 = receiveBTCModule(ctbModule); !r2)
            {
                return tl::unexpected(r2.error());
            }
        }
        else
        {
            CTBNonModule ctbNonModule;
            ctbNonModule.logicalName = logicalName;
            ctbNonModule.isHeaderUnit = type == FileType::HEADER_UNIT;
            if (const auto &r2 = receiveBTCNonModule(ctbNonModule); !r2)
            {
                return tl::unexpected(r2.error());
            }
        }

        return responses.at(logicalName);
    }
    else
    {
        return it->second;
    }
}

tl::expected<void, std::string> IPCManagerCompiler::sendCTBLastMessage(const uint32_t fileSize) const
{
    std::string buffer = getBufferWithType(CTB::LAST_MESSAGE);
    writeUInt32(buffer, fileSize);
    writeUInt32(buffer, buffer.size());
    buffer.append(delimiter, strlen(delimiter));
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<void, std::string> IPCManagerCompiler::sendCTBLastMessage(const std::string &bmiFile,
                                                                       const std::string &filePath) const
{
#ifdef _WIN32
    const HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_READ | GENERIC_WRITE,
                                     0, // no sharing during setup
                                     nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        return tl::unexpected(getErrorString());
    }

    // mappingName is needed as the Windows kernel object names can't have \\ in them.
    const uint64_t hash = rapidhash(filePath.data(), filePath.size());
    char mappingName[17];
    static constexpr char hex[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++)
    {
        const uint8_t byte = hash >> (56 - i * 8) & 0xFF;
        mappingName[i * 2] = hex[byte >> 4];
        mappingName[i * 2 + 1] = hex[byte & 0xF];
    }
    mappingName[16] = '\0';

    LARGE_INTEGER fileSize;
    fileSize.QuadPart = bmiFile.size();
    // 3) Create a RW mapping of that file:
    const HANDLE hMap =
        CreateFileMappingA(hFile, nullptr, PAGE_READWRITE, fileSize.HighPart, fileSize.LowPart, mappingName);
    if (!hMap)
    {
        return tl::unexpected(getErrorString());
    }

    void *pView = MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, bmiFile.size());
    if (!pView)
    {
        return tl::unexpected(getErrorString());
    }

    memcpy(pView, bmiFile.c_str(), bmiFile.size());

    if (!FlushViewOfFile(pView, bmiFile.size()))
    {
        return tl::unexpected(getErrorString());
    }

    UnmapViewOfFile(pView);
    CloseHandle(hFile);

    if (const auto &r = sendCTBLastMessage(fileSize.QuadPart); !r)
    {
        return tl::unexpected(r.error());
    }

    // Build-system will send the BTCLastMessage after it has created the BMI file-mapping. Compiler process can not
    // exit before that.
    if (const auto &r = receiveBTCLastMessage(); !r)
    {
        return tl::unexpected(r.error());
    }

    CloseHandle(hMap);
#else

    const uint64_t fileSize = bmiFile.size();
    // 1. Open & size
    const int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd == -1)
    {
        return tl::unexpected(getErrorString());
    }
    if (ftruncate(fd, fileSize) == -1)
    {
        return tl::unexpected(getErrorString());
    }

    // 2. Map for write
    void *mapping = mmap(nullptr, fileSize, PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED)
    {
        return tl::unexpected(getErrorString());
    }

    // 3. We no longer need the FD
    close(fd);

    memcpy(mapping, bmiFile.data(), bmiFile.size());

    // 4. Flush to disk synchronously
    if (msync(mapping, fileSize, MS_SYNC) == -1)
    {
        return tl::unexpected(getErrorString());
    }

    if (const auto &r = sendCTBLastMessage(fileSize); !r)
    {
        return tl::unexpected(r.error());
    }

    // Build-system will send the BTCLastMessage after it has created the BMI file-mapping. Compiler process can not
    // exit before that.
    if (const auto &r = receiveBTCLastMessage(); !r)
    {
        return tl::unexpected(r.error());
    }
    munmap(mapping, fileSize);

#endif

    return {};
}

tl::expected<Mapping, std::string> IPCManagerCompiler::readSharedMemoryBMIFile(const BMIFile &file)
{
    Mapping f{};
#ifdef _WIN32

    // mappingName is needed as the Windows kernel object names can't have \\ in them.
    const uint64_t hash = rapidhash(file.filePath.data(), file.filePath.size());
    char mappingName[17];
    static constexpr char hex[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++)
    {
        const uint8_t byte = hash >> (56 - i * 8) & 0xFF;
        mappingName[i * 2] = hex[byte >> 4];
        mappingName[i * 2 + 1] = hex[byte & 0xF];
    }
    mappingName[16] = '\0';

    // 1) Open the existing file‐mapping object (must have been created by another process)
    const HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, // read‐only access
                                            FALSE,         // do not inherit a handle
                                            mappingName    // name of mapping
    );

    if (mapping == nullptr)
    {
        return tl::unexpected(getErrorString());
    }

    // 2) Map a view of the file into our address space
    const LPVOID view = MapViewOfFile(mapping,       // handle to mapping object
                                      FILE_MAP_READ, // read‐only view
                                      0,             // file offset high
                                      0,             // file offset low
                                      file.fileSize  // number of bytes to map (0 maps the whole file)
    );

    if (view == nullptr)
    {
        return tl::unexpected(getErrorString());
    }

    f.mapping = mapping;
    f.view = view;
    f.file = {static_cast<char *>(view), file.fileSize};
#else
    const int fd = open(file.filePath.data(), O_RDONLY);
    if (fd == -1)
    {
        return tl::unexpected(getErrorString());
    }
    void *mapping = mmap(nullptr, file.fileSize, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);

    if (close(fd) == -1)
    {
        return tl::unexpected(getErrorString());
    }

    if (mapping == MAP_FAILED)
    {
        return tl::unexpected(getErrorString());
    }

    f.file = {static_cast<char *>(mapping), file.fileSize};
#endif
    return f;
}

tl::expected<void, std::string> IPCManagerCompiler::closeBMIFileMapping(const Mapping &processMappingOfBMIFile)
{
#ifdef _WIN32
    UnmapViewOfFile(processMappingOfBMIFile.view);
    CloseHandle(processMappingOfBMIFile.mapping);
#else
    if (munmap((void *)processMappingOfBMIFile.file.data(), processMappingOfBMIFile.file.size()) == -1)
    {
        return tl::unexpected(getErrorString());
    }
#endif
    return {};
}

bool operator==(const CTBNonModule &lhs, const CTBNonModule &rhs)
{
    return lhs.isHeaderUnit == rhs.isHeaderUnit && lhs.logicalName == rhs.logicalName;
}
} // namespace P2978