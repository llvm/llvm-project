
#include "clang/IPC2978/IPCManagerCompiler.hpp"
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/rapidhash.h"

#include <string>

#ifdef _WIN32
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

using std::string;

namespace N2978
{

tl::expected<IPCManagerCompiler, string> makeIPCManagerCompiler(string BMIIfHeaderUnitObjOtherwisePath)
{
#ifdef _WIN32
    BMIIfHeaderUnitObjOtherwisePath = R"(\\.\pipe\)" + BMIIfHeaderUnitObjOtherwisePath;
    HANDLE hPipe = CreateFileA(BMIIfHeaderUnitObjOtherwisePath.data(), // pipe name
                               GENERIC_READ |                          // read and write access
                                   GENERIC_WRITE,
                               0,             // no sharing
                               nullptr,       // default security attributes
                               OPEN_EXISTING, // opens existing pipe
                               0,             // default attributes
                               nullptr);      // no template file

    // Break if the pipe handle is valid.

    if (hPipe == INVALID_HANDLE_VALUE)
    {
        return tl::unexpected(getErrorString());
    }

    return IPCManagerCompiler(hPipe);
#else

    const int fdSocket = socket(AF_UNIX, SOCK_STREAM, 0);

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
    string prependDir = "/tmp/";
    const uint64_t hash = rapidhash(BMIIfHeaderUnitObjOtherwisePath.c_str(), BMIIfHeaderUnitObjOtherwisePath.size());
    prependDir.append(to16charHexString(hash));
    std::copy(prependDir.begin(), prependDir.end(), addr.sun_path);

    if (connect(fdSocket, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == -1)
    {
        return tl::unexpected(getErrorString());
    }
    return IPCManagerCompiler(fdSocket);

#endif
}

#ifdef _WIN32
IPCManagerCompiler::IPCManagerCompiler(void *hPipe_)
{
    hPipe = hPipe_;
}
#else
IPCManagerCompiler::IPCManagerCompiler(const int fdSocket_)
{
    fdSocket = fdSocket_;
}
#endif

tl::expected<void, string> IPCManagerCompiler::receiveBTCLastMessage() const
{
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

    if (buffer[0] != static_cast<char>(true))
    {
        return tl::unexpected(getErrorString(ErrorCategory::INCORRECT_BTC_LAST_MESSAGE));
    }

    if (constexpr uint32_t bytesProcessed = 1; bytesRead != bytesProcessed)
    {
        return tl::unexpected(getErrorString(bytesRead, bytesProcessed));
    }

    return {};
}

tl::expected<BTCModule, string> IPCManagerCompiler::receiveBTCModule(const CTBModule &moduleName) const
{

    // raise(SIGTRAP); // At the location of the BP.
    vector<char> buffer = getBufferWithType(CTB::MODULE);
    writeString(buffer, moduleName.moduleName);
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }

    return receiveMessage<BTCModule>();
}

tl::expected<BTCNonModule, string> IPCManagerCompiler::receiveBTCNonModule(const CTBNonModule &nonModule) const
{
    vector<char> buffer = getBufferWithType(CTB::NON_MODULE);
    buffer.emplace_back(nonModule.isHeaderUnit);
    writeString(buffer, nonModule.logicalName);
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return receiveMessage<BTCNonModule>();
}

tl::expected<void, string> IPCManagerCompiler::sendCTBLastMessage(const CTBLastMessage &lastMessage) const
{
    vector<char> buffer = getBufferWithType(CTB::LAST_MESSAGE);
    buffer.emplace_back(lastMessage.errorOccurred);
    writeString(buffer, lastMessage.output);
    writeString(buffer, lastMessage.errorOutput);
    writeString(buffer, lastMessage.logicalName);
    writeUInt32(buffer, lastMessage.fileSize);
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }
    return {};
}

tl::expected<void, string> IPCManagerCompiler::sendCTBLastMessage(const CTBLastMessage &lastMessage,
                                                                  const string &bmiFile, const string &filePath) const
{
#ifdef _WIN32
    const HANDLE hFile = CreateFileA(filePath.c_str(), GENERIC_READ | GENERIC_WRITE,
                                     0, // no sharing during setup
                                     nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        return tl::unexpected(getErrorString());
    }

    LARGE_INTEGER fileSize;
    fileSize.QuadPart = bmiFile.size();
    // 3) Create a RW mapping of that file:
    const HANDLE hMap =
        CreateFileMappingA(hFile, nullptr, PAGE_READWRITE, fileSize.HighPart, fileSize.LowPart, filePath.c_str());
    if (!hMap)
    {
        CloseHandle(hFile);
        return tl::unexpected(getErrorString());
    }

    void *pView = MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, bmiFile.size());
    if (!pView)
    {
        CloseHandle(hFile);
        CloseHandle(hMap);
        return tl::unexpected(getErrorString());
    }

    memcpy(pView, bmiFile.c_str(), bmiFile.size());

    if (!FlushViewOfFile(pView, bmiFile.size()))
    {
        UnmapViewOfFile(pView);
        CloseHandle(hFile);
        CloseHandle(hMap);
        return tl::unexpected(getErrorString());
    }

    UnmapViewOfFile(pView);
    CloseHandle(hFile);

    if (const auto &r = sendCTBLastMessage(lastMessage); !r)
    {
        return tl::unexpected(r.error());
    }

    if (lastMessage.errorOccurred == EXIT_SUCCESS)
    {
        if (const auto &r = receiveBTCLastMessage(); !r)
        {
            return tl::unexpected(r.error());
        }
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

    if (const auto &r = sendCTBLastMessage(lastMessage); !r)
    {
        return tl::unexpected(r.error());
    }

    if (lastMessage.errorOccurred == EXIT_SUCCESS)
    {
        if (const auto &r = receiveBTCLastMessage(); !r)
        {
            return tl::unexpected(r.error());
        }
    }

    munmap(mapping, fileSize);
#endif

    return {};
}

tl::expected<ProcessMappingOfBMIFile, string> IPCManagerCompiler::readSharedMemoryBMIFile(const BMIFile &file)
{
    ProcessMappingOfBMIFile f{};
#ifdef _WIN32
    // 1) Open the existing file‐mapping object (must have been created by another process)
    const HANDLE mapping = OpenFileMappingA(FILE_MAP_READ,       // read‐only access
                                            FALSE,               // do not inherit a handle
                                            file.filePath.data() // name of mapping
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
        CloseHandle(mapping);
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

    f.mapping = mapping;
    f.mappingSize = file.fileSize;
    f.file = {static_cast<char *>(mapping), file.fileSize};
#endif
    return f;
}

tl::expected<void, string> IPCManagerCompiler::closeBMIFileMapping(
    const ProcessMappingOfBMIFile &processMappingOfBMIFile)
{
#ifdef _WIN32
    UnmapViewOfFile(processMappingOfBMIFile.view);
    CloseHandle(processMappingOfBMIFile.mapping);
#else
    if (munmap(processMappingOfBMIFile.mapping, processMappingOfBMIFile.mappingSize) == -1)
    {
        return tl::unexpected(getErrorString());
    }
#endif
    return {};
}

void IPCManagerCompiler::closeConnection() const
{
#ifdef _WIN32
    CloseHandle(hPipe);
#else
    close(fdSocket);
#endif
}

bool operator==(const CTBNonModule &lhs, const CTBNonModule &rhs)
{
    return lhs.isHeaderUnit == rhs.isHeaderUnit && lhs.logicalName == rhs.logicalName;
}

uint64_t CTBNonModuleHash::operator()(const CTBNonModule &ctb) const
{
    const_cast<char &>(ctb.logicalName[ctb.logicalName.size()]) = ctb.isHeaderUnit;
    const uint64_t hash = rapidhash(ctb.logicalName.data(), ctb.logicalName.size() + 1);
    const_cast<char &>(ctb.logicalName[ctb.logicalName.size()]) = '\0';
    return hash;
}

} // namespace N2978