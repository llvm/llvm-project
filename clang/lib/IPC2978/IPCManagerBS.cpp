#include "clang/IPC2978/IPCManagerBS.hpp"
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/expected.hpp"
#include <string>
#include <sys/stat.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include "clang/IPC2978/rapidhash.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace N2978
{

tl::expected<IPCManagerBS, std::string> makeIPCManagerBS(std::string BMIIfHeaderUnitObjOtherwisePath)
{
#ifdef _WIN32
    BMIIfHeaderUnitObjOtherwisePath = R"(\\.\pipe\)" + BMIIfHeaderUnitObjOtherwisePath;
    void *hPipe = CreateNamedPipeA(BMIIfHeaderUnitObjOtherwisePath.c_str(), // pipe name
                                   PIPE_ACCESS_DUPLEX |                     // read/write access
                                       FILE_FLAG_FIRST_PIPE_INSTANCE,       // overlapped mode
                                   PIPE_TYPE_MESSAGE |                      // message-type pipe
                                       PIPE_READMODE_MESSAGE |              // message read mode
                                       PIPE_WAIT,                           // blocking mode
                                   1,                                       // unlimited instances
                                   BUFFERSIZE * sizeof(TCHAR),              // output buffer size
                                   BUFFERSIZE * sizeof(TCHAR),              // input buffer size
                                   PIPE_TIMEOUT,                            // client time-out
                                   nullptr);                                // default security attributes
    if (hPipe == INVALID_HANDLE_VALUE)
    {
        return tl::unexpected(getErrorString());
    }
    return IPCManagerBS(hPipe);

#else

    // Named Pipes are used but Unix Domain sockets could have been used as well. The tradeoff is that a file is created
    // and there needs to be bind, listen, accept calls which means that an extra fd is created is temporarily on the
    // server side. it can be closed immediately after.

    const int fdSocket = socket(AF_UNIX, SOCK_STREAM, 0);

    // Create server socket
    if (fdSocket == -1)
    {
        return tl::unexpected(getErrorString());
    }

    // Prepare address structure
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    // We use file hash to make a file path smaller, since there is a limit of NAME_MAX that is generally 108 bytes.
    // TODO
    // Have an option to receive this path in constructor to make it compatible with Android and IOS.
    std::string prependDir = "/tmp/";
    const uint64_t hash = rapidhash(BMIIfHeaderUnitObjOtherwisePath.c_str(), BMIIfHeaderUnitObjOtherwisePath.size());
    prependDir.append(to16charHexString(hash));
    std::copy(prependDir.begin(), prependDir.end(), addr.sun_path);

    // Remove any existing socket
    unlink(prependDir.c_str());

    // Bind socket to the file system path
    if (bind(fdSocket, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == -1)
    {
        return tl::unexpected(getErrorString());
    }
    if (chmod(prependDir.c_str(), 0666) == -1)
    {
        return tl::unexpected(getErrorString());
    }

    // Listen for incoming connections
    if (listen(fdSocket, 1) == -1)
    {
        close(fdSocket);
        return tl::unexpected(getErrorString());
    }

    return IPCManagerBS(fdSocket);
#endif
}

#ifdef _WIN32
IPCManagerBS::IPCManagerBS(void *hPipe_)
{
    hPipe = hPipe_;
}
#else
IPCManagerBS::IPCManagerBS(const int fdSocket_)
{
    fdSocket = fdSocket_;
}
#endif

tl::expected<void, std::string> IPCManagerBS::receiveMessage(char (&ctbBuffer)[320], CTB &messageType) const
{
    if (!connectedToCompiler)
    {
#ifdef _WIN32
        if (!ConnectNamedPipe(hPipe, nullptr))
        {
            // Is the client already connected?
            if (GetLastError() != ERROR_PIPE_CONNECTED)
            {

                DWORD bytesAvail = 0;
                DWORD bytesLeftThisMessage = 0;

                // PeekNamedPipe returns FALSE if pipe is disconnected
                if (PeekNamedPipe(hPipe, nullptr, 0, nullptr, &bytesAvail, &bytesLeftThisMessage))
                {
                    // compiler process ended and has left a message for us.
                }
                else
                {
                    return tl::unexpected(getErrorString());
                }
            }
        }
#else
        const int fd = accept(fdSocket, nullptr, nullptr);
        close(fdSocket);
        if (fd == -1)
        {
            return tl::unexpected(getErrorString());
        }
        const_cast<int &>(fdSocket) = fd;
#endif
        const_cast<bool &>(connectedToCompiler) = true;
    }
    //    raise(SIGTRAP); // At the location of the BP.

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

    uint32_t bytesProcessed = 1;

    // read call fails if zero byte is read, so safe to process 1 byte
    switch (static_cast<CTB>(buffer[0]))
    {

    case CTB::MODULE: {

        const auto &r = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }

        messageType = CTB::MODULE;
        getInitializedObjectFromBuffer<CTBModule>(ctbBuffer).moduleName = *r;
    }

    break;

    case CTB::NON_MODULE: {

        const auto &r = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r)
        {
            return tl::unexpected(r.error());
        }

        const auto &r2 = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!r2)
        {
            return tl::unexpected(r.error());
        }

        messageType = CTB::NON_MODULE;
        auto &[isHeaderUnit, str] = getInitializedObjectFromBuffer<CTBNonModule>(ctbBuffer);
        isHeaderUnit = *r;
        str = *r2;
    }

    break;

    case CTB::LAST_MESSAGE: {

        const auto &exitStatusExpected = readBoolFromPipe(buffer, bytesRead, bytesProcessed);
        if (!exitStatusExpected)
        {
            return tl::unexpected(exitStatusExpected.error());
        }

        const auto &outputExpected = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!outputExpected)
        {
            return tl::unexpected(outputExpected.error());
        }

        const auto &errorOutputExpected = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!errorOutputExpected)
        {
            return tl::unexpected(errorOutputExpected.error());
        }

        const auto &logicalNameExpected = readStringFromPipe(buffer, bytesRead, bytesProcessed);
        if (!logicalNameExpected)
        {
            return tl::unexpected(logicalNameExpected.error());
        }

        const auto &fileSizeExpected = readUInt32FromPipe(buffer, bytesRead, bytesProcessed);
        if (!fileSizeExpected)
        {
            return tl::unexpected(fileSizeExpected.error());
        }

        messageType = CTB::LAST_MESSAGE;

        auto &[exitStatus, output, errorOutput, logicalName, fileSize] =
            getInitializedObjectFromBuffer<CTBLastMessage>(ctbBuffer);

        exitStatus = *exitStatusExpected;
        output = *outputExpected;
        errorOutput = *errorOutputExpected;
        logicalName = *logicalNameExpected;
        fileSize = *fileSizeExpected;
    }
    break;
    }

    if (bytesRead != bytesProcessed)
    {
        return tl::unexpected(getErrorString(bytesRead, bytesProcessed));
    }

    return {};
}

tl::expected<void, std::string> IPCManagerBS::sendMessage(const BTCModule &moduleFile) const
{
    std::vector<char> buffer;
    writeProcessMappingOfBMIFile(buffer, moduleFile.requested);
    buffer.emplace_back(moduleFile.isSystem);
    writeVectorOfModuleDep(buffer, moduleFile.modDeps);
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<void, std::string> IPCManagerBS::sendMessage(const BTCNonModule &nonModule) const
{
    std::vector<char> buffer;
    buffer.emplace_back(nonModule.isHeaderUnit);
    buffer.emplace_back(nonModule.isSystem);
    writeString(buffer, nonModule.filePath);
    writeUInt32(buffer, nonModule.fileSize);
    writeVectorOfStrings(buffer, nonModule.logicalNames);
    writeVectorOfHeaderFiles(buffer, nonModule.headerFiles);
    writeVectorOfHuDeps(buffer, nonModule.huDeps);
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<void, std::string> IPCManagerBS::sendMessage(const BTCLastMessage &) const
{
    std::vector<char> buffer;
    buffer.emplace_back(true);
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<ProcessMappingOfBMIFile, std::string> IPCManagerBS::createSharedMemoryBMIFile(BMIFile &bmiFile)
{
    ProcessMappingOfBMIFile sharedFile{};
#ifdef _WIN32

    std::string mappingName = bmiFile.filePath;
    for (char &c : mappingName)
    {
        if (c == '\\')
        {
            c = '/';
        }
    }

    if (bmiFile.fileSize == UINT32_MAX)
    {
        const HANDLE hFile = CreateFileA(bmiFile.filePath.c_str(), GENERIC_READ,
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
            CreateFileMappingA(hFile, nullptr, PAGE_READONLY, fileSize.HighPart, fileSize.LowPart, mappingName.c_str());

        CloseHandle(hFile);

        if (!sharedFile.mapping)
        {
            return tl::unexpected(getErrorString());
        }

        bmiFile.fileSize = fileSize.QuadPart;
        return sharedFile;
    }

    // 1) Open the existing file‐mapping object (must have been created by another process)
    sharedFile.mapping = OpenFileMappingA(FILE_MAP_READ,      // read‐only access
                                          FALSE,              // do not inherit handle
                                          mappingName.c_str() // name of mapping
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

tl::expected<void, std::string> IPCManagerBS::closeBMIFileMapping(
    const ProcessMappingOfBMIFile &processMappingOfBMIFile)
{
#ifdef _WIN32
    CloseHandle(processMappingOfBMIFile.mapping);
#else
    if (munmap((void *)processMappingOfBMIFile.file.data(), processMappingOfBMIFile.file.size()) == -1)
    {
        return tl::unexpected(getErrorString());
    }
#endif
    return {};
}

void IPCManagerBS::closeConnection() const
{
#ifdef _WIN32
    CloseHandle(hPipe);
#else
    close(fdSocket);
#endif
}

} // namespace N2978