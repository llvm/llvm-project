
#include "clang/IPC2978/IPCManagerCompiler.hpp"
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define TRY_READ(var, func, ...)                                                                                       \
    const auto &var = func(__VA_ARGS__);                                                                               \
    if (!var)                                                                                                          \
    {                                                                                                                  \
        return Error{var.error()};                                                                                     \
    }

#define TRY_READ_VAL(var, func, ...)                                                                                   \
    const auto &var##_result = func(__VA_ARGS__);                                                                      \
    if (!var##_result)                                                                                                 \
    {                                                                                                                  \
        return Error{var##_result.error()};                                                                            \
    }                                                                                                                  \
    auto &var = *var##_result;

namespace P2978
{

static bool endsWith(const std::string_view str, const std::string_view suffix)
{
    if (suffix.size() > str.size())
    {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

Result<std::string_view> IPCManagerCompiler::readInternal(char (&buffer)[4096]) const
{
    std::string *output = nullptr;
    while (true)
    {
        uint64_t bytesRead = 0;
#ifdef _WIN32
        DWORD readCount = 0;
        const bool success = ReadFile(GetStdHandle(STD_INPUT_HANDLE), buffer, sizeof(buffer), &readCount, nullptr);

        bytesRead = readCount;
        if (const DWORD lastError = GetLastError(); !success && lastError != ERROR_MORE_DATA)
        {
            return Error{getErrorString()};
        }

#else
        bytesRead = read(STDIN_FILENO, buffer, 4096);
        if (bytesRead == UINT64_MAX)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return Error{getErrorString()};
        }

#endif
        if (!bytesRead)
        {
            return Error{getErrorString(ErrorCategory::READ_FILE_ZERO_BYTES_READ)};
        }

        if (!output)
        {
            output = allocations.emplace_back(std::make_unique<std::string>()).get();
        }

        output->append(buffer, bytesRead);

        // Retain the complete response: cached names and paths borrow its bytes.
        if (endsWith(*output, delimiter))
        {
            return std::string_view{output->data(), output->size() - strlen(delimiter)};
        }
    }
}

Result<void> IPCManagerCompiler::writeInternal(const std::string_view buffer) const
{
#ifdef _WIN32
    return writeAll(GetStdHandle(STD_OUTPUT_HANDLE), buffer);
#else
    return writeAll(STDOUT_FILENO, buffer.data(), buffer.size());
#endif
}

Result<Response> IPCManagerCompiler::readBMIResponse(const std::string_view message, uint64_t &bytesRead,
                                                     const FileType type, const bool isSystem)
{
    TRY_READ_VAL(filePath, readPath, message, bytesRead);
    TRY_READ_VAL(contents, loadBMIContents, filePath);
    return Response{filePath, contents, type, isSystem};
}

Result<std::string_view> IPCManagerCompiler::loadBMIContents(const std::string_view filePath)
{
    if (const auto it = bmiContentsByPath.find(filePath); it != bmiContentsByPath.end())
    {
        return it->second;
    }

    TRY_READ_VAL(contents, mapBMIFile, filePath);
    bmiContentsByPath.emplace(filePath, contents);
    return contents;
}

Result<std::string_view> IPCManagerCompiler::findBMIContents(const std::string_view filePath) const
{
    const auto it = bmiContentsByPath.find(filePath);
    if (it == bmiContentsByPath.end())
    {
        return Error{std::string("BMI was not supplied by the build system: ") + std::string(filePath)};
    }
    return it->second;
}

Result<void> IPCManagerCompiler::readLogicalNames(const std::string_view message, uint64_t &bytesRead,
                                                  const Response &response)
{
    TRY_READ_VAL(logicalNamesSize, readUInt32, message, bytesRead);
    for (uint64_t i = 0; i < logicalNamesSize; ++i)
    {
        TRY_READ_VAL(logicalName, readString, message, bytesRead);
        responses.emplace(logicalName, response);
    }
    return {};
}

Result<void> IPCManagerCompiler::receiveBTCModule(const CTBModule &moduleName)
{
    std::string buffer = getBufferWithType(CTB::MODULE);
    writeString(buffer, moduleName.moduleName);
    writeUInt32(buffer, buffer.size());
    buffer.append(delimiter, strlen(delimiter));
    if (const auto &r = writeInternal(buffer); !r)
    {
        return Error{r.error()};
    }

    char stackBuffer[4096];
    auto received = readInternal(stackBuffer);

    if (!received)
    {
        return Error{received.error()};
    }
    const std::string_view message = *received;

    uint64_t bytesRead = 0;

    TRY_READ_VAL(requested, readBMIResponse, message, bytesRead, FileType::MODULE);
    TRY_READ_VAL(isSystem, readBool, message, bytesRead);

    const auto &str = allocations.emplace_back(std::make_unique<std::string>(moduleName.moduleName));
    responses.emplace(*str, Response{requested.filePath, requested.bmiContents, FileType::MODULE, isSystem});

    TRY_READ_VAL(modDepsSize, readUInt32, message, bytesRead);

    for (uint64_t i = 0; i < modDepsSize; ++i)
    {
        TRY_READ_VAL(isHeaderUnit, readBool, message, bytesRead);
        const FileType type = isHeaderUnit ? FileType::HEADER_UNIT : FileType::MODULE;
        TRY_READ_VAL(modDep, readBMIResponse, message, bytesRead, type);
        TRY_READ_VAL(isSystemDep, readBool, message, bytesRead);
        TRY_READ(aliases, readLogicalNames, message, bytesRead,
                 (Response{modDep.filePath, modDep.bmiContents, type, isSystemDep}));
    }

    if (message.size() != bytesRead)
    {
        return Error{getErrorString(ErrorCategory::PARSING_ERROR)};
    }
    return {};
}

Result<void> IPCManagerCompiler::receiveBTCNonModule(const CTBNonModule &nonModule)
{
    std::string buffer = getBufferWithType(CTB::NON_MODULE);
    buffer.push_back(nonModule.isHeaderUnit);
    writeString(buffer, nonModule.logicalName);
    writeUInt32(buffer, buffer.size());
    buffer.append(delimiter, strlen(delimiter));
    if (const auto &r = writeInternal(buffer); !r)
    {
        return Error{r.error()};
    }

    char stackBuffer[4096];
    auto received = readInternal(stackBuffer);

    if (!received)
    {
        return Error{received.error()};
    }

    std::string_view readCompilerMessage = *received;
    uint64_t bytesRead = 0;

    TRY_READ_VAL(isHeaderUnit, readBool, readCompilerMessage, bytesRead);
    TRY_READ_VAL(isSystem, readBool, readCompilerMessage, bytesRead);
    TRY_READ_VAL(headerFilesSize, readUInt32, readCompilerMessage, bytesRead);

    for (uint64_t i = 0; i < headerFilesSize; ++i)
    {
        TRY_READ_VAL(logicalName, readString, readCompilerMessage, bytesRead);
        TRY_READ_VAL(filePath, readPath, readCompilerMessage, bytesRead);
        TRY_READ_VAL(isSystemHeaderFile, readBool, readCompilerMessage, bytesRead);

        responses.emplace(logicalName, Response{filePath, {}, FileType::HEADER_FILE, isSystemHeaderFile});
    }

    const auto &str = allocations.emplace_back(std::make_unique<std::string>(nonModule.logicalName));
    if (!isHeaderUnit)
    {
        TRY_READ_VAL(filePath, readPath, readCompilerMessage, bytesRead);
        responses.emplace(*str, Response{filePath, {}, FileType::HEADER_FILE, isSystem});
        if (readCompilerMessage.size() != bytesRead)
        {
            return Error{getErrorString(ErrorCategory::PARSING_ERROR)};
        }
        return {};
    }

    TRY_READ_VAL(file, readBMIResponse, readCompilerMessage, bytesRead, FileType::HEADER_UNIT, isSystem);
    responses.emplace(*str, file);

    TRY_READ(logicalNames, readLogicalNames, readCompilerMessage, bytesRead, file);

    TRY_READ_VAL(huDepsSize, readUInt32, readCompilerMessage, bytesRead);
    for (uint64_t i = 0; i < huDepsSize; ++i)
    {
        TRY_READ_VAL(huDep, readBMIResponse, readCompilerMessage, bytesRead, FileType::HEADER_UNIT);
        TRY_READ_VAL(huDepIsSystem, readBool, readCompilerMessage, bytesRead);
        TRY_READ(aliases, readLogicalNames, readCompilerMessage, bytesRead,
                 (Response{huDep.filePath, huDep.bmiContents, FileType::HEADER_UNIT, huDepIsSystem}));
    }

    if (readCompilerMessage.size() != bytesRead)
    {
        return Error{getErrorString(ErrorCategory::PARSING_ERROR)};
    }
    return {};
}

Result<Response> IPCManagerCompiler::findResponse(const std::string_view logicalName, const FileType type)
{
#ifdef _WIN32
    std::string logicalName2{logicalName};
    if (type != FileType::MODULE)
    {
        for (char &c : logicalName2)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
#else
    const std::string_view logicalName2 = logicalName;
#endif

    // Include translation permits a header unit to satisfy a textual-header request.
    if (const auto &it = responses.find(logicalName2);
        it == responses.end() ||
        (it->second.type != type && (it->second.type != FileType::HEADER_UNIT || type != FileType::HEADER_FILE)))
    {
        if (isMocking)
        {
            return Error{"Could not find entry in mocking-mode"};
        }

        if (type == FileType::MODULE)
        {
            CTBModule ctbModule;
            ctbModule.moduleName = logicalName2;
            if (const auto &r2 = receiveBTCModule(ctbModule); !r2)
            {
                return Error{r2.error()};
            }
        }
        else
        {
            CTBNonModule ctbNonModule;
            ctbNonModule.logicalName = logicalName2;
            ctbNonModule.isHeaderUnit = type == FileType::HEADER_UNIT;
            if (const auto &r2 = receiveBTCNonModule(ctbNonModule); !r2)
            {
                return Error{r2.error()};
            }
        }

        return responses.at(logicalName2);
    }
    else
    {
        return it->second;
    }
}

static Result<std::string> fileToString(const std::string_view fileName)
{
    std::ifstream file(std::string(fileName), std::ios::binary);
    if (!file)
    {
        return Error{std::string("Could not open IPC mock file: ") + std::string(fileName)};
    }
    std::string fileBuffer{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (file.bad())
    {
        return Error{std::string("Could not read IPC mock file: ") + std::string(fileName)};
    }
    return fileBuffer;
}

Result<void> IPCManagerCompiler::readEntriesFromFile(const std::string_view filePath)
{
    // Neither a second mock nor a switch from live IPC may replace storage borrowed by cached entries.
    if (isMocking || !responses.empty() || !allocations.empty() || !bmiContentsByPath.empty())
    {
        return Error{"IPC mock dependencies can only be loaded once on a fresh manager"};
    }
    isMocking = true;
    auto contents = fileToString(filePath);
    if (!contents)
    {
        return Error{contents.error()};
    }
    scanCacheFileData = std::move(*contents);

    uint64_t bytesRead = 0;
    TRY_READ_VAL(entriesSize, readUInt32, scanCacheFileData, bytesRead);
    for (uint64_t i = 0; i < entriesSize; ++i)
    {
        TRY_READ_VAL(responseKey, readString, scanCacheFileData, bytesRead);

        TRY_READ_VAL(valueFilePath, readPath, scanCacheFileData, bytesRead);
        TRY_READ_VAL(fileType, readUInt8, scanCacheFileData, bytesRead);
        if (fileType > static_cast<uint8_t>(FileType::HEADER_FILE))
        {
            return Error{getErrorString(ErrorCategory::PARSING_ERROR)};
        }
        TRY_READ_VAL(isSystem, readBool, scanCacheFileData, bytesRead);

        const FileType type = static_cast<FileType>(fileType);
        std::string_view contents;
        if (type != FileType::HEADER_FILE)
        {
            TRY_READ_VAL(mapped, loadBMIContents, valueFilePath);
            contents = mapped;
        }
        responses.emplace(responseKey, Response{valueFilePath, contents, type, isSystem});
    }

    if (bytesRead != scanCacheFileData.size())
    {
        return Error{getErrorString(ErrorCategory::PARSING_ERROR)};
    }

    mockFilePath = filePath;
    return {};
}

Result<std::string_view> IPCManagerCompiler::mapBMIFile(const std::string_view filePath)
{
    // Own a terminated path; callers may supply arbitrary string_views.
    const std::string path(filePath);
#ifdef _WIN32
    const HANDLE handle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
    {
        return Error{getErrorString()};
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(handle, &size))
    {
        const std::string error = getErrorString();
        CloseHandle(handle);
        return Error{error};
    }
    if (size.QuadPart <= 0 || static_cast<uint64_t>(size.QuadPart) > (std::numeric_limits<size_t>::max)())
    {
        CloseHandle(handle);
        return Error{std::string("Invalid BMI file size: ") + path};
    }

    // No name or build-system-owned object is needed. All readers open the completed file.
    const HANDLE mapping = CreateFileMappingA(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping)
    {
        const std::string error = getErrorString();
        CloseHandle(handle);
        return Error{error};
    }
    CloseHandle(handle);

    const void *view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view)
    {
        const std::string error = getErrorString();
        CloseHandle(mapping);
        return Error{error};
    }
    // The view holds its own reference to the mapping object. Keep the view until process exit.
    CloseHandle(mapping);
    return std::string_view{static_cast<const char *>(view), static_cast<size_t>(size.QuadPart)};
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
    {
        return Error{getErrorString()};
    }

    struct stat st;
    if (fstat(fd, &st) == -1)
    {
        const std::string error = getErrorString();
        close(fd);
        return Error{error};
    }
    if (st.st_size <= 0 || static_cast<uint64_t>(st.st_size) > (std::numeric_limits<size_t>::max)())
    {
        close(fd);
        return Error{std::string("Invalid BMI file size: ") + path};
    }
    const size_t size = static_cast<size_t>(st.st_size);
    int flags = MAP_SHARED;
#ifdef MAP_POPULATE
    flags |= MAP_POPULATE;
#endif
    void *view = mmap(nullptr, size, PROT_READ, flags, fd, 0);
    if (view == MAP_FAILED)
    {
        const std::string error = getErrorString();
        close(fd);
        return Error{error};
    }
    close(fd);
    // Closing the descriptor leaves the view valid; process exit releases the mapping.
    return std::string_view{static_cast<const char *>(view), size};
#endif
}

} // namespace P2978
