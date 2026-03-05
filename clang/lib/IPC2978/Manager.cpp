
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/expected.hpp"

#ifdef _WIN32
#include <Windows.h>
#else
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace P2978
{

std::string getErrorString()
{
#ifdef _WIN32
    const DWORD err = GetLastError();

    char *msg_buf;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                   err, MAKELANGID(LANG_ENGLISH, SUBLANG_DEFAULT), reinterpret_cast<char *>(&msg_buf), 0, nullptr);

    if (msg_buf == nullptr)
    {
        char fallback_msg[128] = {};
        snprintf(fallback_msg, sizeof(fallback_msg), "GetLastError() = %ld", err);
        return fallback_msg;
    }

    std::string msg = msg_buf;
    LocalFree(msg_buf);
    return msg;
#else
    return {std::strerror(errno)};
#endif
}

std::string getErrorString(const uint32_t bytesRead_, const uint32_t bytesProcessed_)
{
    return "Error: Bytes Readd vs Bytes Processed Mismatch.\nBytes Read: " + std::to_string(bytesRead_) +
           ", Bytes Processed: " + std::to_string(bytesProcessed_);
}

std::string getErrorString(const ErrorCategory errorCategory_)
{
    std::string errorString;

    switch (errorCategory_)
    {
    case ErrorCategory::PARSING_ERROR:
        errorString = "P2978 Message Parsing Error.";
        break;
    case ErrorCategory::READ_FILE_ZERO_BYTES_READ:
        errorString = "Error: ReadFile Zero Bytes Read.";
        break;
    case ErrorCategory::INCORRECT_BTC_LAST_MESSAGE:
        errorString = "Error: Incorrect BTC Last Message.";
        break;
    case ErrorCategory::UNKNOWN_CTB_TYPE:
        errorString = "Error: Unknown CTB message received.";
        break;
    case ErrorCategory::NONE:
        std::string str = __FILE__;
        str += ':';
        str += __LINE__;
        errorString = "P2978 IPC API internal error" + str;
        break;
    }

    return errorString;
}

#ifndef _WIN32
tl::expected<void, std::string> Manager::writeAll(const int fd, const char *buffer, const uint32_t count)
{
    uint32_t bytesWritten = 0;

    while (bytesWritten != count)
    {
        const int32_t result = write(fd, buffer + bytesWritten, count - bytesWritten);
        if (result == -1)
        {
            if (errno == EINTR)
            {
                // Interrupted by signal: retry
                continue;
            }
            return tl::unexpected(getErrorString());
        }
        if (result == 0)
        {
            // According to POSIX, write() returning 0 is only valid for count == 0
            return tl::unexpected(getErrorString());
        }
        bytesWritten += result;
    }

    return {};
}
#endif

std::string Manager::getBufferWithType(CTB type)
{
    std::string buffer;
    buffer.push_back(static_cast<uint8_t>(type));
    return buffer;
}

void Manager::writeUInt32(std::string &buffer, const uint32_t value)
{
    const auto ptr = reinterpret_cast<const char *>(&value);
    buffer.append(ptr, ptr + 4);
}

void Manager::writeString(std::string &buffer, const std::string_view &str)
{
    writeUInt32(buffer, str.size());
    buffer.append(str.begin(), str.end()); // Insert all characters
}

void Manager::writePath(std::string &buffer, const std::string_view &str)
{
    writeUInt32(buffer, str.size());
    buffer.append(str.begin(), str.end()); // Insert all characters
    buffer.push_back('\0');
}

void Manager::writeBMIFile(std::string &buffer, const BMIFile &file)
{
    writePath(buffer, file.filePath);
    writeUInt32(buffer, file.fileSize);
}

void Manager::writeModuleDep(std::string &buffer, const ModuleDep &dep)
{
    buffer.push_back(dep.isHeaderUnit);
    writeBMIFile(buffer, dep.file);
    buffer.push_back(dep.isSystem);
    writeVectorOfStrings(buffer, dep.logicalNames);
}

void Manager::writeHuDep(std::string &buffer, const HuDep &dep)
{
    writeBMIFile(buffer, dep.file);
    buffer.push_back(dep.isSystem);
    writeVectorOfStrings(buffer, dep.logicalNames);
}

void Manager::writeHeaderFile(std::string &buffer, const HeaderFile &dep)
{
    writeString(buffer, dep.logicalName);
    writePath(buffer, dep.filePath);
    buffer.push_back(dep.isSystem);
}

void Manager::writeVectorOfStrings(std::string &buffer, const std::vector<std::string_view> &strs)
{
    writeUInt32(buffer, strs.size());
    for (const std::string_view &str : strs)
    {
        writeString(buffer, str);
    }
}

void Manager::writeVectorOfProcessMappingOfBMIFiles(std::string &buffer, const std::vector<BMIFile> &files)
{
    writeUInt32(buffer, files.size());
    for (const BMIFile &file : files)
    {
        writeBMIFile(buffer, file);
    }
}

void Manager::writeVectorOfModuleDep(std::string &buffer, const std::vector<ModuleDep> &deps)
{
    writeUInt32(buffer, deps.size());
    for (const ModuleDep &dep : deps)
    {
        writeModuleDep(buffer, dep);
    }
}

void Manager::writeVectorOfHuDeps(std::string &buffer, const std::vector<HuDep> &deps)
{
    writeUInt32(buffer, deps.size());
    for (const HuDep &dep : deps)
    {
        writeHuDep(buffer, dep);
    }
}

void Manager::writeVectorOfHeaderFiles(std::string &buffer, const std::vector<HeaderFile> &headerFiles)
{
    writeUInt32(buffer, headerFiles.size());
    for (const HeaderFile &headerFile : headerFiles)
    {
        writeHeaderFile(buffer, headerFile);
    }
}

tl::expected<bool, std::string> Manager::readBool(const std::string_view message, uint32_t &bytesRead)
{
    if (bytesRead + 1 > message.size())
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }
    bool result = *reinterpret_cast<const bool *>(message.data() + bytesRead);
    bytesRead += 1;
    return result;
}

tl::expected<uint32_t, std::string> Manager::readUInt32(const std::string_view message, uint32_t &bytesRead)
{
    if (bytesRead + 4 > message.size())
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }
    uint32_t result = *reinterpret_cast<const uint32_t *>(message.data() + bytesRead);
    bytesRead += 4;
    return result;
}

tl::expected<std::string_view, std::string> Manager::readString(const std::string_view message, uint32_t &bytesRead)
{
    auto r = readUInt32(message, bytesRead);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    const uint32_t stringSize = *r;
    if (bytesRead + stringSize > message.size())
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }
    std::string_view result = {message.data() + bytesRead, stringSize};
    bytesRead += stringSize;
    return result;
}

tl::expected<std::string_view, std::string> Manager::readPath(const std::string_view message, uint32_t &bytesRead)
{
    auto r = readUInt32(message, bytesRead);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    const uint32_t stringSize = *r;
    if (bytesRead + stringSize > message.size())
    {
        return tl::unexpected(getErrorString(ErrorCategory::PARSING_ERROR));
    }
    std::string_view result = {message.data() + bytesRead, stringSize};
    bytesRead += stringSize;
    // This string is followed by \0
    bytesRead += 1;
    return result;
}

} // namespace P2978