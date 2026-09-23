// Response sending for the simulated build systems in the protocol and Clang tests.
#ifndef IPC2978_TEST_BUILD_SYSTEM_HPP
#define IPC2978_TEST_BUILD_SYSTEM_HPP

#ifdef IS_THIS_CLANG_REPO
#include "clang/IPC2978/Manager.hpp"
#else
#include "Manager.hpp"
#endif
#include <cstring>

namespace ipc2978_test
{
class TestBuildSystem
{
    // Borrowed pipe descriptor on Unix, or HANDLE encoded as uint64_t on Windows.
    uint64_t writeFd;

    P2978::Result<void> writeInternal(std::string_view buffer) const
    {
#ifdef _WIN32
        return P2978::Manager::writeAll(reinterpret_cast<void *>(writeFd), buffer);
#else
        return P2978::Manager::writeAll(static_cast<int>(writeFd), buffer.data(), buffer.size());
#endif
    }

  public:
    // The test process retains ownership of the compiler's input pipe.
    explicit TestBuildSystem(uint64_t writeFd_) : writeFd(writeFd_)
    {
    }

    [[nodiscard]] P2978::Result<void> sendMessage(const P2978::BTCModule &moduleFile) const
    {
        std::string buffer;
        P2978::Manager::writePath(buffer, moduleFile.filePath);
        buffer.push_back(moduleFile.isSystem);
        P2978::Manager::writeVectorOfModuleDep(buffer, moduleFile.modDeps);
        buffer.append(P2978::delimiter, std::strlen(P2978::delimiter));
        return writeInternal(buffer);
    }

    [[nodiscard]] P2978::Result<void> sendMessage(const P2978::BTCNonModule &nonModule) const
    {
        std::string buffer;
        buffer.push_back(nonModule.isHeaderUnit);
        buffer.push_back(nonModule.isSystem);
        P2978::Manager::writeVectorOfHeaderFiles(buffer, nonModule.headerFiles);
        P2978::Manager::writePath(buffer, nonModule.filePath);
        if (nonModule.isHeaderUnit)
        {
            P2978::Manager::writeVectorOfStrings(buffer, nonModule.logicalNames);
            P2978::Manager::writeVectorOfHuDeps(buffer, nonModule.huDeps);
        }
        buffer.append(P2978::delimiter, std::strlen(P2978::delimiter));
        return writeInternal(buffer);
    }
};
} // namespace ipc2978_test
#endif // IPC2978_TEST_BUILD_SYSTEM_HPP
