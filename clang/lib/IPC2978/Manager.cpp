
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

namespace N2978
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
        errorString = "N2978 IPC API internal error" + str;
        break;
    }

    return errorString;
}

tl::expected<uint32_t, std::string> Manager::readInternal(char (&buffer)[BUFFERSIZE]) const
{
    int32_t bytesRead;

#ifdef _WIN32
    const bool success = ReadFile(hPipe,               // pipe handle
                                  buffer,              // buffer to receive reply
                                  BUFFERSIZE,          // size of buffer
                                  LPDWORD(&bytesRead), // number of bytes read
                                  nullptr);            // not overlapped

    if (const uint32_t lastError = GetLastError(); !success && lastError != ERROR_MORE_DATA)
    {
        return tl::unexpected(getErrorString());
    }

#else
    bytesRead = read(fdSocket, buffer, BUFFERSIZE);
    if (bytesRead == -1)
    {
        return tl::unexpected(getErrorString());
    }
#endif

    if (!bytesRead)
    {
        return tl::unexpected(getErrorString(ErrorCategory::READ_FILE_ZERO_BYTES_READ));
    }

    return bytesRead;
}

#ifndef _WIN32
tl::expected<void, std::string> writeAll(const int fd, const char *buffer, const uint32_t count)
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

tl::expected<void, std::string> Manager::writeInternal(const std::vector<char> &buffer) const
{
#ifdef _WIN32
    const bool success = WriteFile(hPipe,         // pipe handle
                                   buffer.data(), // message
                                   buffer.size(), // message length
                                   nullptr,       // bytes written
                                   nullptr);      // not overlapped
    if (!success)
    {
        return tl::unexpected(getErrorString());
    }
#else
    if (const auto &r = writeAll(fdSocket, buffer.data(), buffer.size()); !r)
    {
        return tl::unexpected(r.error());
    }
#endif
    return {};
}

std::vector<char> Manager::getBufferWithType(CTB type)
{
    std::vector<char> buffer;
    buffer.emplace_back(static_cast<uint8_t>(type));
    return buffer;
}

void Manager::writeUInt32(std::vector<char> &buffer, const uint32_t value)
{
    const auto ptr = reinterpret_cast<const char *>(&value);
    buffer.insert(buffer.end(), ptr, ptr + 4);
}

void Manager::writeString(std::vector<char> &buffer, const std::string &str)
{
    writeUInt32(buffer, str.size());
    buffer.insert(buffer.end(), str.begin(), str.end()); // Insert all characters
}

void Manager::writeProcessMappingOfBMIFile(std::vector<char> &buffer, const BMIFile &file)
{
    writeString(buffer, file.filePath);
    writeUInt32(buffer, file.fileSize);
}

void Manager::writeModuleDep(std::vector<char> &buffer, const ModuleDep &dep)
{
    buffer.emplace_back(dep.isHeaderUnit);
    writeProcessMappingOfBMIFile(buffer, dep.file);
    writeVectorOfStrings(buffer, dep.logicalNames);
    buffer.emplace_back(dep.isSystem);
}

void Manager::writeHuDep(std::vector<char> &buffer, const HuDep &dep)
{
    writeProcessMappingOfBMIFile(buffer, dep.file);
    writeVectorOfStrings(buffer, dep.logicalNames);
    buffer.emplace_back(dep.isSystem);
}

void Manager::writeHeaderFile(std::vector<char> &buffer, const HeaderFile &dep)
{
    writeString(buffer, dep.logicalName);
    writeString(buffer, dep.filePath);
    buffer.emplace_back(dep.isSystem);
}

void Manager::writeVectorOfStrings(std::vector<char> &buffer, const std::vector<std::string> &strs)
{
    writeUInt32(buffer, strs.size());
    for (const std::string &str : strs)
    {
        writeString(buffer, str);
    }
}

void Manager::writeVectorOfProcessMappingOfBMIFiles(std::vector<char> &buffer, const std::vector<BMIFile> &files)
{
    writeUInt32(buffer, files.size());
    for (const BMIFile &file : files)
    {
        writeProcessMappingOfBMIFile(buffer, file);
    }
}

void Manager::writeVectorOfModuleDep(std::vector<char> &buffer, const std::vector<ModuleDep> &deps)
{
    writeUInt32(buffer, deps.size());
    for (const ModuleDep &dep : deps)
    {
        writeModuleDep(buffer, dep);
    }
}

void Manager::writeVectorOfHuDeps(std::vector<char> &buffer, const std::vector<HuDep> &deps)
{
    writeUInt32(buffer, deps.size());
    for (const HuDep &dep : deps)
    {
        writeHuDep(buffer, dep);
    }
}

void Manager::writeVectorOfHeaderFiles(std::vector<char> &buffer, const std::vector<HeaderFile> &headerFiles)
{
    writeUInt32(buffer, headerFiles.size());
    for (const HeaderFile &headerFile : headerFiles)
    {
        writeHeaderFile(buffer, headerFile);
    }
}

tl::expected<bool, std::string> Manager::readBoolFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                          uint32_t &bytesProcessed) const
{
    bool result;
    const auto &r =
        readNumberOfBytes(reinterpret_cast<char *>(&result), sizeof(result), buffer, bytesRead, bytesProcessed);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    return result;
}

tl::expected<uint32_t, std::string> Manager::readUInt32FromPipe(char (&buffer)[4096], uint32_t &bytesRead,
                                                                uint32_t &bytesProcessed) const
{
    uint32_t size;
    if (const auto &r = readNumberOfBytes(reinterpret_cast<char *>(&size), 4, buffer, bytesRead, bytesProcessed); !r)
    {
        return tl::unexpected(r.error());
    }
    return size;
}

tl::expected<std::string, std::string> Manager::readStringFromPipe(char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                                   uint32_t &bytesProcessed) const
{
    auto r = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    const uint32_t stringSize = *r;
    std::string str(stringSize, 'a');
    if (const auto &r2 = readNumberOfBytes(str.data(), stringSize, buffer, bytesRead, bytesProcessed); !r2)
    {
        return tl::unexpected(r2.error());
    }
    return str;
}

tl::expected<BMIFile, std::string> Manager::readProcessMappingOfBMIFileFromPipe(char (&buffer)[4096],
                                                                                uint32_t &bytesRead,
                                                                                uint32_t &bytesProcessed) const
{
    const auto &r = readStringFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    const auto &r2 = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);

    BMIFile file;
    file.filePath = *r;
    file.fileSize = *r2;
    return file;
}

tl::expected<std::vector<std::string>, std::string> Manager::readVectorOfStringFromPipe(char (&buffer)[BUFFERSIZE],
                                                                                        uint32_t &bytesRead,
                                                                                        uint32_t &bytesProcessed) const
{
    const auto &r = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
    if (!r)
    {
        return tl::unexpected(r.error());
    }
    const uint32_t vectorSize = *r;
    std::vector<std::string> vec;
    vec.reserve(vectorSize);
    for (uint32_t i = 0; i < vectorSize; ++i)
    {
        const auto &r2 = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r2)
        {
            return tl::unexpected(r2.error());
        }
        vec.emplace_back(*r2);
    }
    return vec;
}

tl::expected<ModuleDep, std::string> Manager::readModuleDepFromPipe(char (&buffer)[4096], uint32_t &bytesRead,
                                                                    uint32_t &bytesProcessed) const
{
    const auto &r = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r)
    {
        return tl::unexpected(r.error());
    }

    const auto &r2 = readProcessMappingOfBMIFileFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r2)
    {
        return tl::unexpected(r2.error());
    }

    const auto &r3 = readVectorOfStringFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r3)
    {
        return tl::unexpected(r3.error());
    }

    const auto &r4 = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r4)
    {
        return tl::unexpected(r4.error());
    }

    ModuleDep modDep;

    modDep.isHeaderUnit = *r;
    modDep.file = *r2;
    modDep.logicalNames = *r3;
    modDep.isSystem = *r4;

    return modDep;
}

tl::expected<std::vector<ModuleDep>, std::string> Manager::readVectorOfModuleDepFromPipe(char (&buffer)[4096],
                                                                                         uint32_t &bytesRead,
                                                                                         uint32_t &bytesProcessed) const
{
    const auto &vectorSize = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
    if (!vectorSize)
    {
        return tl::unexpected(vectorSize.error());
    }

    std::vector<ModuleDep> vec;
    vec.reserve(*vectorSize);
    for (uint32_t i = 0; i < *vectorSize; ++i)
    {
        const auto &r = readModuleDepFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }

        vec.emplace_back(*r);
    }
    return vec;
}

tl::expected<HuDep, std::string> Manager::readHuDepFromPipe(char (&buffer)[4096], uint32_t &bytesRead,
                                                            uint32_t &bytesProcessed) const
{
    const auto &r = readProcessMappingOfBMIFileFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r)
    {
        return tl::unexpected(r.error());
    }

    const auto &r2 = readVectorOfStringFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r2)
    {
        return tl::unexpected(r2.error());
    }

    const auto &r3 = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
    if (!r3)
    {
        return tl::unexpected(r3.error());
    }

    HuDep huDep;
    huDep.file = *r;
    huDep.logicalNames = *r2;
    huDep.isSystem = *r3;
    return huDep;
}

tl::expected<std::vector<HuDep>, std::string> Manager::readVectorOfHuDepFromPipe(char (&buffer)[4096],
                                                                                 uint32_t &bytesRead,
                                                                                 uint32_t &bytesProcessed) const
{
    const auto &vectorSize = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
    if (!vectorSize)
    {
        return tl::unexpected(vectorSize.error());
    }

    std::vector<HuDep> vec;
    vec.reserve(*vectorSize);
    for (uint32_t i = 0; i < *vectorSize; ++i)
    {
        const auto &r = readHuDepFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }
        vec.emplace_back(*r);
    }
    return vec;
}

tl::expected<HeaderFile, std::string> Manager::readHeaderFileFromPipe(char (&buffer)[4096], uint32_t &bytesRead,
                                                                      uint32_t &bytesProcessed) const
{

    const auto &r = readStringFromPipe(buffer, bytesRead, bytesProcessed);
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

    HeaderFile hf;
    hf.logicalName = *r;
    hf.filePath = *r2;
    hf.isSystem = *r3;
    return hf;
}

tl::expected<std::vector<HeaderFile>, std::string> Manager::readVectorOfHeaderFileFromPipe(
    char (&buffer)[4096], uint32_t &bytesRead, uint32_t &bytesProcessed) const
{
    const auto &vectorSize = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
    if (!vectorSize)
    {
        return tl::unexpected(vectorSize.error());
    }

    std::vector<HeaderFile> vec;
    vec.reserve(*vectorSize);
    for (uint32_t i = 0; i < *vectorSize; ++i)
    {
        const auto &r = readHeaderFileFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }
        vec.emplace_back(*r);
    }
    return vec;
}

tl::expected<void, std::string> Manager::readNumberOfBytes(char *output, const uint32_t size,
                                                           char (&buffer)[BUFFERSIZE], uint32_t &bytesRead,
                                                           uint32_t &bytesProcessed) const
{
    uint32_t pendingSize = size;
    uint32_t offset = 0;
    while (true)
    {
        const uint32_t bytesAvailable = bytesRead - bytesProcessed;
        if (bytesAvailable >= pendingSize)
        {
            memcpy(output + offset, buffer + bytesProcessed, pendingSize);
            bytesProcessed += pendingSize;
            break;
        }

        if (bytesAvailable)
        {
            memcpy(output + offset, buffer + bytesProcessed, bytesAvailable);
            offset += bytesAvailable;
            pendingSize -= bytesAvailable;
        }

        bytesProcessed = 0;
        if (const auto &r = readInternal(buffer); r)
        {
            bytesRead = *r;
        }
        else
        {
            return tl::unexpected(r.error());
        }
    }
    return {};
}
} // namespace N2978