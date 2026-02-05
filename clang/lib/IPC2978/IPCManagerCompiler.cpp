
#include "clang/IPC2978/IPCManagerCompiler.hpp"
#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"
#include "clang/IPC2978/rapidhash.h"

#include <string>
#include <utility>

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

namespace N2978
{

tl::expected<IPCManagerCompiler, std::string> makeIPCManagerCompiler(std::string BMIIfHeaderUnitObjOtherwisePath)
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
    std::string prependDir = "/tmp/";
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

Response::Response(std::string filePath_, const ProcessMappingOfBMIFile &mapping_, const FileType type_,
                   const bool isSystem_)
    : filePath(std::move(filePath_)), mapping(mapping_), type(type_), isSystem(isSystem_)
{
}

tl::expected<void, std::string> IPCManagerCompiler::receiveBTCLastMessage() const
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

    // The BTCLastMessage must be 1 byte of true signaling that build-system has successfully created a shared memory
    // mapping of the BMI file.
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

tl::expected<BTCModule, std::string> IPCManagerCompiler::receiveBTCModule(const CTBModule &moduleName)
{
    std::vector<char> buffer = getBufferWithType(CTB::MODULE);
    writeString(buffer, moduleName.moduleName);
    // This call sends the CTBModule to the build-system.
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }

    auto received = receiveMessage<BTCModule>();

    if (!received)
    {
        return tl::unexpected(received.error());
    }

    auto &[f, isSystem, deps] = received.value();
    if (const auto &r = readSharedMemoryBMIFile(f); r)
    {
        filePathProcessMapping.emplace(f.filePath, r.value());
        responses.emplace(moduleName.moduleName,
                          Response(std::move(f.filePath), r.value(), FileType::MODULE, isSystem));
    }
    else
    {
        return tl::unexpected(r.error());
    }

    // Build-system will also send all the dependencies of this module.
    for (auto &[isHeaderUnit, file, logicalNames, isSystem2] : deps)
    {
        if (isHeaderUnit)
        {
            // logicalNames include all the include-names of the composing-headers of the big header-unit dependency.
            // These are mapped to the dependency big-hu in IPCManagerCompiler::responses cache. e.g. if the module
            // "Cat" depends on the "string" which is compiled as big-hu "stl", then build-system will send all the
            // composing include-names like "vector" etc. So if we any of these is requested later, response cache will
            // have them mapped to the big-hu BMI "stl".
            if (const auto &r = readSharedMemoryBMIFile(file); r)
            {
                filePathProcessMapping.emplace(file.filePath, r.value());
                for (std::string s : logicalNames)
                {
                    responses.emplace(std::move(s),
                                      Response(file.filePath, r.value(), FileType::HEADER_UNIT, isSystem2));
                }
            }
            else
            {
                return tl::unexpected(r.error());
            }
        }
        else
        {
            if (const auto &r = readSharedMemoryBMIFile(file); r)
            {
                filePathProcessMapping.emplace(file.filePath, r.value());
                // logicalNames[0] is the module logicalName of the dependency module.
                responses.emplace(logicalNames[0],
                                  Response(std::move(file.filePath), r.value(), FileType::MODULE, isSystem2));
            }
            else
            {
                return tl::unexpected(r.error());
            }
        }
    }
    return received;
}

tl::expected<BTCNonModule, std::string> IPCManagerCompiler::receiveBTCNonModule(const CTBNonModule &nonModule)
{
    std::vector<char> buffer = getBufferWithType(CTB::NON_MODULE);
    buffer.emplace_back(nonModule.isHeaderUnit);
    writeString(buffer, nonModule.logicalName);
    // This call sends the CTBNonModule to the build-system.
    if (const auto &r = writeInternal(buffer); !r)
    {
        return tl::unexpected(r.error());
    }

    auto received = receiveMessage<BTCNonModule>();

    if (!received)
    {
        return tl::unexpected(received.error());
    }

    auto &[isHeaderUnit, isSystem, filePath, fileSize, logicalNames, headerFiles, huDeps] = received.value();

    // the requested header-file or header-unit
    BMIFile f;
    f.filePath = std::move(filePath);
    f.fileSize = fileSize;

    if (isHeaderUnit)
    {
        if (const auto &r = readSharedMemoryBMIFile(f); r)
        {
            filePathProcessMapping.emplace(f.filePath, r.value());
            for (std::string &h : logicalNames)
            {
                // if our request was compiled as big-hu, then the logicalNames include all the composing include-names
                // of the big-hu. e.g. if we requested "string" which was compiled as big-hu "stl", this array will
                // include "vector", "algorithm" etc. These will be mapped to the same BMIFile. This reduces no. of
                // messages.
                responses.emplace(std::move(h), Response(f.filePath, r.value(), FileType::HEADER_UNIT, isSystem));
            }
        }
        else
        {
            return tl::unexpected(r.error());
        }
    }

    // All the hu-deps of this header-unit.
    for (auto &[file, logicalHUDep, isSystem2] : huDeps)
    {
        if (const auto &r = readSharedMemoryBMIFile(file); r)
        {
            filePathProcessMapping.emplace(file.filePath, r.value());
            // All the composing include-names of the dependency if any. These will be mapped to the dependency big hu.
            for (std::string &l : logicalHUDep)
            {
                responses.emplace(std::move(l), Response(file.filePath, r.value(), FileType::HEADER_UNIT, isSystem2));
            }
        }
        else
        {
            return tl::unexpected(r.error());
        }
    }

    // if we are compiling a big hu, only then the following is received. It is only received in first request. It is
    // the list of all the composing header-files.
    for (auto &[logicalName, headerFilePath, isSystem2] : headerFiles)
    {
        BMIFile headerBMI;
        headerBMI.filePath = std::move(headerFilePath);
        responses.emplace(std::move(logicalName),
                          Response(std::move(headerBMI.filePath), {}, FileType::HEADER_FILE, isSystem2));
    }

    responses.emplace(
        nonModule.logicalName,
        Response(std::move(f.filePath), {}, isHeaderUnit ? FileType::HEADER_UNIT : FileType::HEADER_FILE, isSystem));

    return received;
}

tl::expected<Response, std::string> IPCManagerCompiler::findResponse(std::string logicalName, const FileType type)
{
#ifdef _WIN32
    if (type != FileType::MODULE)
    {
        for (char &c : logicalName)
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

tl::expected<void, std::string> IPCManagerCompiler::sendCTBLastMessage() const
{
    std::vector<char> buffer = getBufferWithType(CTB::LAST_MESSAGE);
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

    LARGE_INTEGER fileSize;
    fileSize.QuadPart = bmiFile.size();
    // 3) Create a RW mapping of that file:
    const HANDLE hMap =
        CreateFileMappingA(hFile, nullptr, PAGE_READWRITE, fileSize.HighPart, fileSize.LowPart, filePath.c_str());
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

    if (const auto &r = sendCTBLastMessage(); !r)
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

    if (const auto &r = sendCTBLastMessage(); !r)
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

tl::expected<ProcessMappingOfBMIFile, std::string> IPCManagerCompiler::readSharedMemoryBMIFile(const BMIFile &file)
{
    ProcessMappingOfBMIFile f{};
#ifdef _WIN32
    std::string mappingName = file.filePath;
    for (char &c : mappingName)
    {
        if (c == '\\')
        {
            c = '/';
        }
    }
    // 1) Open the existing file‐mapping object (must have been created by another process)
    const HANDLE mapping = OpenFileMappingA(FILE_MAP_READ,      // read‐only access
                                            FALSE,              // do not inherit a handle
                                            mappingName.c_str() // name of mapping
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

tl::expected<void, std::string> IPCManagerCompiler::closeBMIFileMapping(
    const ProcessMappingOfBMIFile &processMappingOfBMIFile)
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
} // namespace N2978