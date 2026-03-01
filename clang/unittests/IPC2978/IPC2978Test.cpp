//===- unittests/IPC2978/IPC2978Test.cpp - Tests IPC2978 Support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The following line is uncommented by clang/lib/IPC2978/setup.py for clang/unittests/IPC2978/IPC2978.cpp

#define IS_THIS_CLANG_REPO
#include <cstring>
#include <iostream>
#ifdef IS_THIS_CLANG_REPO
#include "clang/IPC2978/IPCManagerBS.hpp"
#include "gtest/gtest.h"
#else
#include "IPCManagerBS.hpp"
#include "Testing.hpp"
#endif

#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/epoll.h>
#include <sys/wait.h>
#include <unistd.h>
#include <wordexp.h>
#endif

using namespace std::filesystem;
using namespace P2978;
using namespace std;

#ifdef _WIN32
#define CLANG_CMD ".\\clang.exe"
#else
#define CLANG_CMD "./clang"
#endif

namespace
{

void exitFailure(const string &str)
{
#ifdef IS_THIS_CLANG_REPO
    FAIL() << str + '\n';
#else
    std::cerr << str << std::endl;
    exit(EXIT_FAILURE);
#endif
}

uint64_t createMultiplex()
{
#ifdef _WIN32
    HANDLE iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, // handle to associate
                                         nullptr,              // existing IOCP handle
                                         0,                    // completion key (use pipe handle)
                                         0                     // number of concurrent threads (0 = default)
    );
    if (iocp == nullptr)
    {
        exitFailure(getErrorString());
    }
    return reinterpret_cast<uint64_t>(iocp);
#else
    return epoll_create1(0);
#endif
}

struct RunCommand
{
    uint64_t pid;
    uint64_t readPipe;
    uint64_t writePipe;
    int exitStatus;
    uint64_t startAsyncProcess(const char *command, uint64_t serverFd);
    void reapProcess() const;
};

#ifdef _WIN32

// Copied partially from Ninja
uint64_t RunCommand::startAsyncProcess(const char *command, uint64_t serverFd)
{
    // One BTarget can launch multiple processes so we also append the clock::now().
    const string read_pipe_name = R"(\\.\pipe\read{}{})";
    const string write_pipe_name = R"(\\.\pipe\write{}{})";

    // ===== CREATE READ PIPE (for reading child's stdout/stderr - ASYNC) =====
    HANDLE readPipe_ = CreateNamedPipeA(read_pipe_name.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                                        PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, 0, 0, INFINITE, NULL);
    if (readPipe_ == INVALID_HANDLE_VALUE)
    {
        exitFailure(getErrorString());
    }

    // Register read pipe with IOCP for async operations
    if (!CreateIoCompletionPort(readPipe_, (HANDLE)serverFd, (ULONG_PTR)readPipe_, 0))
    {
        exitFailure(getErrorString());
    }

    OVERLAPPED readOverlappedIO = {};
    if (!ConnectNamedPipe(readPipe_, &readOverlappedIO) && GetLastError() != ERROR_IO_PENDING)
    {
        exitFailure(getErrorString());
    }

    // Get the write end for child's stdout/stderr
    HANDLE outputWriteHandle = CreateFileA(read_pipe_name.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (outputWriteHandle == INVALID_HANDLE_VALUE)
    {
        exitFailure(getErrorString());
    }

    {
        char buffer[4096];
        DWORD bytesRead = 0;
        OVERLAPPED overlapped = {0};

        // Wait for the read to complete.
        ULONG_PTR completionKey = 0;
        LPOVERLAPPED completedOverlapped = nullptr;
        if (!GetQueuedCompletionStatus((HANDLE)serverFd, &bytesRead, &completionKey, &completedOverlapped, INFINITE))
        {
            exitFailure(getErrorString());
        }
    }

    HANDLE childStdoutPipe;
    if (!DuplicateHandle(GetCurrentProcess(), outputWriteHandle, GetCurrentProcess(), &childStdoutPipe, 0, TRUE,
                         DUPLICATE_SAME_ACCESS))
    {
        exitFailure(getErrorString());
    }
    CloseHandle(outputWriteHandle);

    // ===== CREATE WRITE PIPE (for writing to child's stdin - SYNC) =====
    // Note: No FILE_FLAG_OVERLAPPED - this makes it synchronous
    HANDLE writePipe_ = CreateNamedPipeA(write_pipe_name.c_str(),
                                         PIPE_ACCESS_OUTBOUND, // No FILE_FLAG_OVERLAPPED
                                         PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, 0, 0, INFINITE, NULL);
    if (writePipe_ == INVALID_HANDLE_VALUE)
    {
        exitFailure(getErrorString());
    }

    // Get the read end for child's stdin
    HANDLE inputReadHandle = CreateFileA(write_pipe_name.c_str(), GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (inputReadHandle == INVALID_HANDLE_VALUE)
    {
        exitFailure(getErrorString());
    }

    HANDLE childStdinPipe;
    if (!DuplicateHandle(GetCurrentProcess(), inputReadHandle, GetCurrentProcess(), &childStdinPipe, 0, TRUE,
                         DUPLICATE_SAME_ACCESS))
    {
        exitFailure(getErrorString());
    }
    CloseHandle(inputReadHandle);

    STARTUPINFOA startup_info;
    memset(&startup_info, 0, sizeof(startup_info));
    startup_info.cb = sizeof(STARTUPINFO);

    bool use_console_ = false;
    if (!use_console_)
    {
        startup_info.dwFlags = STARTF_USESTDHANDLES;
        startup_info.hStdInput = childStdinPipe;   // Child reads from this
        startup_info.hStdOutput = childStdoutPipe; // Child writes to this
        startup_info.hStdError = childStdoutPipe;  // Child writes to this
    }

    PROCESS_INFORMATION process_info;
    memset(&process_info, 0, sizeof(process_info));

    // Ninja handles ctrl-c, except for subprocesses in console pools.
    DWORD process_flags = use_console_ ? 0 : CREATE_NEW_PROCESS_GROUP;

    // Do not prepend 'cmd /c' on Windows, this breaks command
    // lines greater than 8,191 chars.
    if (!CreateProcessA(NULL, (char *)command, NULL, NULL,
                        /* inherit handles */ TRUE, process_flags, NULL, NULL, &startup_info, &process_info))
    {
        exitFailure(getErrorString());
    }

    // Close pipe channels only used by the child.
    CloseHandle(childStdoutPipe);
    CloseHandle(childStdinPipe);

    CloseHandle(process_info.hThread);

    readPipe = (uint64_t)readPipe_;   // Parent reads child's output from this (ASYNC)
    writePipe = (uint64_t)writePipe_; // Parent writes to child's input via this (SYNC)
    pid = (uint64_t)process_info.hProcess;

    return readPipe;
}

void RunCommand::reapProcess() const
{
    if (WaitForSingleObject((HANDLE)pid, INFINITE) == WAIT_FAILED)
    {
        exitFailure(getErrorString());
    }

    if (!GetExitCodeProcess((HANDLE)pid, (LPDWORD)&exitStatus))
    {
        exitFailure(getErrorString());
    }

    if (!CloseHandle((HANDLE)pid) || !CloseHandle((HANDLE)readPipe) || !CloseHandle((HANDLE)writePipe))
    {
        exitFailure(getErrorString());
    }
}

#else

uint64_t RunCommand::startAsyncProcess(const char *command, uint64_t serverFd)
{
    // Create pipes for stdout and stderr
    int stdoutPipesLocal[2];
    if (pipe(stdoutPipesLocal) == -1)
    {
        exitFailure(getErrorString());
    }

    // Create pipe for stdin
    int stdinPipesLocal[2];
    if (pipe(stdinPipesLocal) == -1)
    {
        exitFailure(getErrorString());
    }

    readPipe = stdoutPipesLocal[0];
    writePipe = stdinPipesLocal[1];

    pid = fork();
    if (pid == -1)
    {
        exitFailure(getErrorString());
    }
    if (pid == 0)
    {
        // Child process

        // Redirect stdin from the pipe
        dup2(stdinPipesLocal[0], STDIN_FILENO);

        // Redirect stdout and stderr to the pipes
        dup2(stdoutPipesLocal[1], STDOUT_FILENO); // Redirect stdout to stdout_pipe
        dup2(stdoutPipesLocal[1], STDERR_FILENO); // Redirect stderr to stderr_pipe

        // Close unused pipe ends
        close(stdoutPipesLocal[0]);
        close(stdoutPipesLocal[1]);
        close(stdinPipesLocal[0]);
        close(stdinPipesLocal[1]);

        wordexp_t p;
        if (wordexp(command, &p, 0) != 0)
        {
            perror("wordexp");
            _exit(127);
        }

        // p.we_wordv is a NULL-terminated argv suitable for exec*
        char **argv = p.we_wordv;

        // Use execvp so PATH is searched and environment is inherited
        execvp(argv[0], argv);

        // If execvp returns, it failed:
        perror("execvp");
        wordfree(&p);
        _exit(127);
    }

    // Parent process
    // Close unused pipe ends
    close(stdoutPipesLocal[1]);
    close(stdinPipesLocal[0]);
    return readPipe;
}

void RunCommand::reapProcess() const
{
    if (waitpid(pid, const_cast<int *>(&exitStatus), 0) < 0)
    {
        exitFailure(getErrorString());
    }
    if (close(readPipe) == -1)
    {
        exitFailure(getErrorString());
    }
    if (close(writePipe) == -1)
    {
        exitFailure(getErrorString());
    }
}
#endif

string compilerTestPrunedOutput;
uint64_t serverFd;
CTB type;
char buffer[320];
RunCommand runCommand;

bool endsWith(const std::string &str, const std::string &suffix)
{
    if (suffix.size() > str.size())
    {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void readCompilerMessage(const uint64_t serverFd, const uint64_t readFd)
{
#ifdef _WIN32
    HANDLE hIOCP = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(serverFd));
    HANDLE hPipe = reinterpret_cast<HANDLE>(readFd);

    while (true)
    {
        char buffer[4096];
        DWORD bytesRead = 0;
        OVERLAPPED overlapped = {0};

        // Initiate async read. Even if it is completed successfully, we will get the completion packet. We don't get
        // the packet only if it fails immediately with error other than ERROR_IO_PENDING
        BOOL result = ReadFile(hPipe, buffer, sizeof(buffer), &bytesRead, &overlapped);

        DWORD error = GetLastError();
        if (!result && error != ERROR_IO_PENDING)
        {
            if (error == ERROR_BROKEN_PIPE)
            {
                // read complete
                return;
            }
            exitFailure(getErrorString());
        }

        bytesRead = 0;

        // Wait for the read to complete.
        ULONG_PTR completionKey = 0;
        LPOVERLAPPED completedOverlapped = nullptr;

        if (!GetQueuedCompletionStatus(hIOCP, &bytesRead, &completionKey, &completedOverlapped, INFINITE))
        {
            if (GetLastError() == ERROR_BROKEN_PIPE)
            {
                // completed
                return;
            }
            exitFailure(getErrorString());
        }

        // Verify completion is for our pipe
        if (completionKey != (ULONG_PTR)hPipe)
        {
            exitFailure("Unexpected completion key");
        }

        if (bytesRead == 0)
        {
            // completed
            return;
        }

        // Append read data to string
        for (DWORD i = 0; i < bytesRead; ++i)
        {
            compilerTestPrunedOutput.push_back(buffer[i]);
        }

        // Check for terminator
        if (endsWith(compilerTestPrunedOutput, delimiter))
        {
            return;
        }
    }
#else

    epoll_event ev{};
    ev.events = EPOLLIN;
    if (epoll_ctl(serverFd, EPOLL_CTL_ADD, readFd, &ev) == -1)
    {
        exitFailure(getErrorString());
    }

    epoll_wait(serverFd, &ev, 1, -1);
    while (true)
    {
        char buffer[4096];
        const int readCount = read(readFd, buffer, 4096);
        if (readCount == 0)
        {
            return;
        }
        if (readCount == -1)
        {
            exitFailure(getErrorString());
        }
        for (uint32_t i = 0; i < readCount; ++i)
        {
            compilerTestPrunedOutput.push_back(buffer[i]);
        }

        if (endsWith(compilerTestPrunedOutput, delimiter))
        {
            break;
        }
    }

    if (epoll_ctl(serverFd, EPOLL_CTL_DEL, readFd, &ev) == -1)
    {
        exitFailure(getErrorString());
    }
#endif
}
void pruneCompilerOutput(IPCManagerBS &manager)
{
    // Prune the compiler output. and make a new string of the compiler-message output.
    const uint32_t prunedSize = compilerTestPrunedOutput.size();
    if (prunedSize < 4 + strlen(delimiter))
    {
        exitFailure("received string only has delimiter but not the size of payload\n");
    }

    const uint32_t payloadSize =
        *reinterpret_cast<uint32_t *>(compilerTestPrunedOutput.data() + (prunedSize - (4 + strlen(delimiter))));
    const char *payloadStart = compilerTestPrunedOutput.data() + (prunedSize - (4 + strlen(delimiter) + payloadSize));
    if (const auto &r2 = IPCManagerBS::receiveMessage(buffer, type, string_view{payloadStart, payloadSize}); !r2)
    {
        exitFailure(r2.error());
    }
    compilerTestPrunedOutput.resize(prunedSize - (4 + strlen(delimiter) + payloadSize));
}

IPCManagerBS readFirstCompilerStdout(const string_view compileCommand, const bool ctbMessageExpected)
{
    // todo
    //  initialize

    serverFd = createMultiplex();
    runCommand.startAsyncProcess(compileCommand.data(), serverFd);
    IPCManagerBS manager(runCommand.writePipe);

    if (ctbMessageExpected)
    {
        readCompilerMessage(serverFd, runCommand.readPipe);
        if (!endsWith(compilerTestPrunedOutput, delimiter))
        {
            exitFailure("early exit by CompilerTest");
        }
        pruneCompilerOutput(manager);
    }
    return manager;
}

void readCompilerStdout(IPCManagerBS &manager)
{
    readCompilerMessage(serverFd, runCommand.readPipe);
    if (!endsWith(compilerTestPrunedOutput, delimiter))
    {
        exitFailure("early exit by CompilerTest");
    }
    pruneCompilerOutput(manager);
}

void endCompilerTest()
{
    readCompilerMessage(serverFd, runCommand.readPipe);
    runCommand.reapProcess();
    std::cout << compilerTestPrunedOutput << std::endl;
    if (runCommand.exitStatus != EXIT_SUCCESS)
    {
        std::cout << "CompilerTest did not exit successfully. ExitCode is" << runCommand.exitStatus << std::endl;
    }
}

// main.cpp
const string mainDotCpp = R"(
// only one request of Foo will be made as A and big.hpp
// will be provided with it.
import Foo;
import A;
#include "y.hpp"
#include "z.hpp"

int main()
{
    Hello();
    World();
    Foo();
}
)";

// Creates all the input files (source files + pcm files) that are needed for the test.
void setupTest()
{
    //  a.cpp
    const string aDotCpp = R"(
export module A;     // primary module interface unit

export import :B;    // Hello() is visible when importing 'A'.
import :C;           // WorldImpl() is now visible only for 'a.cpp'.
// export import :C; // ERROR: Cannot export a module implementation unit.

// World() is visible by any translation unit importing 'A'.
export char const* World()
{
    return WorldImpl();
}
)";
    // a-b.cpp
    const string aBDotCPP = R"(
export module A:B; // partition module interface unit

// Hello() is visible by any translation unit importing 'A'.
export char const* Hello() { return "Hello"; }
)";

    // a-c.cpp
    const string aCDotCPP = R"(
module A:C; // partition module implementation unit

// WorldImpl() is visible by any module unit of 'A' importing ':C'.
char const* WorldImpl() { return "World"; }
)";

    // m.hpp, n.hpp and o.hpp are to be used as header-units, header-files
    // while x.hpp, y.hpp and z.hpp are to be used as big-hu by include big.hpp

    // m.hpp
    const string mDotHpp = R"(
// this file can not be included without first defining M_HEADER_FILE
// this is to demonstrate difference between header-file and header-unit.
// as macros don't seep into header-units

#ifdef M_HEADER_FILE
inline int m = 5;
#else
fail compilation
#endif
)";

    // n.hpp
    const string nDotHpp = R"(
// should work just fine as macro should not seep in here while inclusion.

#ifdef N_HEADER_FILE
fail compilation
#else
#define M_HEADER_FILE
#include "m.hpp"
inline int n = 5 + m;
#endif

// COMMAND_MACRO should be defined while compiling this.
// however, it should still be fine if it is not defined while compiling
// a file consuming this

#ifndef COMMAND_MACRO
fail compilation
#endif
)";

    // o.hpp
    const string oDotHpp = R"(
// TRANSLATING should be defined if /translateInclude is being used.
// "o.hpp" should still be treated as header-file.

#define M_HEADER_FILE
#include "m.hpp"
#ifdef TRANSLATING
#include "n.hpp"
#else
import "n.hpp";
#endif

inline int o = n + m + 5;
)";

    // x.hpp
    const string xDotHpp = R"(
#ifndef X_HPP
#define X_HPP
inline int x = 5;
#endif
)";

    // y.hpp
    const string yDotHpp = R"(
#ifndef Y_HPP
#define Y_HPP
#include "x.hpp"
inline int y = x + 5;
#endif
)";

    // z.hpp
    const string zDotHpp = R"(
#ifndef Z_HPP
#define Z_HPP
#include "y.hpp"
inline int z = x + y + 5;
#endif
)";

    // big.hpp
    const string bigDotHpp = R"(
#include "x.hpp"
// todo
// following two should not be requested as big.hpp includes the following as well.
#include "y.hpp"
#include "z.hpp"
)";

    // foo.cpp
    const string fooDotCpp = R"(
module;
#include "x.hpp"
#include "z.hpp"
#include "y.hpp"
#include "big.hpp"
export module Foo;
import A;

export void Foo()
{
    Hello();
    World();
    int s = x + y + z;
}
)";

    ofstream("a.cpp") << aDotCpp;
    ofstream("a-b.cpp") << aBDotCPP;
    ofstream("a-c.cpp") << aCDotCPP;
    ofstream("m.hpp") << mDotHpp;
    ofstream("n.hpp") << nDotHpp;
    ofstream("o.hpp") << oDotHpp;
    ofstream("x.hpp") << xDotHpp;
    ofstream("y.hpp") << yDotHpp;
    ofstream("z.hpp") << zDotHpp;
    ofstream("big.hpp") << bigDotHpp;
    ofstream("foo.cpp") << fooDotCpp;
    ofstream("main.cpp") << mainDotCpp;
}

tl::unexpected<string> errorReturn()
{
    return tl::unexpected<string>("IPC2978 Test Error: Wrong Message Received\n");
}

#define CHECK(condition)                                                                                               \
    if (!(condition))                                                                                                  \
    {                                                                                                                  \
        return errorReturn();                                                                                          \
    }

#define CREATE_BMI_MAPPING(varName, filePath)                                                                          \
    Mapping varName;                                                                                                   \
    if (const auto &_r_##varName = IPCManagerBS::createSharedMemoryBMIFile(filePath); _r_##varName)                    \
    {                                                                                                                  \
        varName = *_r_##varName;                                                                                       \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        return tl::unexpected("failed to created bmi mapping" + _r_##varName.error() + "\n");                          \
    }

#define SEND_MESSAGE(message)                                                                                          \
    if (const auto &_r_send_##message = manager.sendMessage(message); !_r_send_##message)                              \
    {                                                                                                                  \
        return tl::unexpected("manager send message failed" + _r_send_##message.error() + "\n");                       \
    }

#define CLOSE_BMI_MAPPING(mapping)                                                                                     \
    if (const auto &_r_close_##mapping = IPCManagerBS::closeBMIFileMapping(mapping); !_r_close_##mapping)              \
    {                                                                                                                  \
        return tl::unexpected("closing bmi-mapping failed");                                                           \
    }

tl::expected<void, string> runTest()
{
    setupTest();

    string str = current_path().string();
#ifdef _WIN32
    for (char &c : str)
    {
        c = tolower(c);
    }
#endif

    path curPath(str);

    string mainFilePath = (curPath / "main .o").string();
    string modFilePath = (curPath / "mod .pcm").string();
    string mod1FilePath = (curPath / "mod1 .pcm").string();
    string mod2FilePath = (curPath / "mod2 .pcm").string();
    string aCObj = (curPath / "a-c .o").string();
    string aCPcm = (curPath / "a-c .pcm").string();
    string aBObj = (curPath / "a-b .o").string();
    string aBPcm = (curPath / "a-b .pcm").string();
    string aObj = (curPath / "a .o").string();
    string aPcm = (curPath / "a .pcm").string();
    string bObj = (curPath / "b .o").string();
    string bPcm = (curPath / "b .pcm").string();
    string mHpp = (curPath / "m.hpp").string();
    string nHpp = (curPath / "n.hpp").string();
    string oHpp = (curPath / "o.hpp").string();
    string nPcm = (curPath / "n .pcm").string();
    string oPcm = (curPath / "o .pcm").string();
    string xHpp = (curPath / "x.hpp").string();
    string yHpp = (curPath / "y.hpp").string();
    string zHpp = (curPath / "z.hpp").string();
    string bigHpp = (curPath / "big.hpp").string();
    string bigPcm = (curPath / "big .pcm").string();
    string fooPcm = (curPath / "foo .pcm").string();
    string fooObj = (curPath / "foo .o").string();
    string mainObj = (curPath / "main .o").string();

    // compiling a-c.cpp
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + aCObj +
                                "\" -noScanIPC -c -xc++-module a-c.cpp -fmodule-output=\"" + aCPcm + "\"";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, false);
        endCompilerTest();
    }

    // compiling a-b.cpp
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + aBObj +
                                "\" -noScanIPC -c -xc++-module a-b.cpp -fmodule-output=\"" + aBPcm + "\"";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, false);
        endCompilerTest();
    }
    // compiling a.cpp
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + aObj +
                                "\" -noScanIPC -c -xc++-module a.cpp -fmodule-output=\"" + aPcm + "\"";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::MODULE)
        const auto &ctbModule = reinterpret_cast<CTBModule &>(buffer);
        CHECK(ctbModule.moduleName == "A:B")

        BMIFile btcModBMI;
        btcModBMI.filePath = aBPcm;
        CREATE_BMI_MAPPING(btcModBmiMapping, btcModBMI)

        BTCModule btcMod;
        btcMod.requested = btcModBMI;

        BMIFile modDepBMI;
        modDepBMI.filePath = aCPcm;
        CREATE_BMI_MAPPING(modDepBmiMapping, modDepBMI)

        ModuleDep modDep;
        modDep.file = modDepBMI;
        modDep.logicalNames.emplace_back("A:C");
        modDep.isHeaderUnit = false;
        btcMod.modDeps.emplace_back(std::move(modDep));

        SEND_MESSAGE(btcMod)

        endCompilerTest();
        CLOSE_BMI_MAPPING(btcModBmiMapping)
        CLOSE_BMI_MAPPING(modDepBmiMapping)
    }

    // compiling n.hpp
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + nPcm +
                                "\" -noScanIPC -xc++-header n.hpp -DCOMMAND_MACRO";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::NON_MODULE)
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(ctbNonModMHpp.logicalName == "m.hpp" || ctbNonModMHpp.isHeaderUnit == false)

        BTCNonModule nonModMPcm;
        nonModMPcm.isHeaderUnit = false;
        nonModMPcm.filePath = mHpp;
        SEND_MESSAGE(nonModMPcm)
        endCompilerTest();
    }

    // compiling o.hpp
    {
        string compileCommand =
            CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + oPcm + "\" -noScanIPC -xc++-header o.hpp";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::NON_MODULE)
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(ctbNonModMHpp.logicalName == "m.hpp" || ctbNonModMHpp.isHeaderUnit == false)

        BTCNonModule nonModMPcm;
        nonModMPcm.isHeaderUnit = false;
        nonModMPcm.filePath = mHpp;
        SEND_MESSAGE(nonModMPcm)

        readCompilerStdout(manager);
        CHECK(type == CTB::NON_MODULE)
        const auto &ctbNonModNHpp = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(ctbNonModNHpp.logicalName == "n.hpp" || ctbNonModNHpp.isHeaderUnit == true)

        BTCNonModule nonModNPcm;
        nonModNPcm.isHeaderUnit = true;

        BMIFile nonModNPcmBmi;
        nonModNPcmBmi.filePath = nPcm;

        CREATE_BMI_MAPPING(nonModNPcmBmiMapping, nonModNPcmBmi)

        nonModNPcm.filePath = nonModNPcmBmi.filePath;
        nonModNPcm.fileSize = nonModNPcmBmi.fileSize;

        SEND_MESSAGE(nonModNPcm)
        endCompilerTest();
        CLOSE_BMI_MAPPING(nonModNPcmBmiMapping)
    }

    // compiling o.hpp with include-translation. BTCNonModule for n.hpp will be received with
    // isHeaderUnit = true.
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + oPcm +
                                "\" -noScanIPC -xc++-header o.hpp -DTRANSLATING";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::NON_MODULE)
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(ctbNonModMHpp.logicalName == "m.hpp" || ctbNonModMHpp.isHeaderUnit == false)

        BTCNonModule nonModMPcm;
        nonModMPcm.isHeaderUnit = false;
        nonModMPcm.filePath = mHpp;
        SEND_MESSAGE(nonModMPcm)

        readCompilerStdout(manager);
        CHECK(type == CTB::NON_MODULE)
        const auto &ctbNonModNHpp = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(ctbNonModNHpp.logicalName == "n.hpp" || ctbNonModNHpp.isHeaderUnit == false)

        BTCNonModule nonModNPcm;

        BMIFile nonModNPcmBmi;
        nonModNPcmBmi.filePath = nPcm;

        CREATE_BMI_MAPPING(nonModNPcmBmiMapping, nonModNPcmBmi)

        nonModNPcm.isHeaderUnit = true;
        nonModNPcm.filePath = nPcm;
        nonModNPcm.fileSize = nonModNPcmBmi.fileSize;

        SEND_MESSAGE(nonModNPcm)
        endCompilerTest();
        CLOSE_BMI_MAPPING(nonModNPcmBmiMapping)
    }

    // compiling big.hpp
    {
        string compileCommand =
            CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + bigPcm + "\" -noScanIPC -xc++-header big.hpp";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::NON_MODULE)
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(ctbNonModMHpp.logicalName == "x.hpp" || ctbNonModMHpp.isHeaderUnit == false)

        BTCNonModule headerFile;
        headerFile.isHeaderUnit = false;
        headerFile.filePath = xHpp;
        HeaderFile yHeaderFile;
        yHeaderFile.logicalName = "y.hpp";
        yHeaderFile.filePath = yHpp;
        yHeaderFile.isSystem = true;
        headerFile.headerFiles.emplace_back(yHeaderFile);
        HeaderFile zHeaderFile;
        zHeaderFile.logicalName = "z.hpp";
        zHeaderFile.filePath = zHpp;
        zHeaderFile.isSystem = true;
        headerFile.headerFiles.emplace_back(zHeaderFile);

        SEND_MESSAGE(headerFile)
        endCompilerTest();
    }

    // compiling foo.cpp
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + fooObj +
                                "\" -noScanIPC -c -xc++-module foo.cpp -fmodule-output=\"" + fooPcm + "\"";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::NON_MODULE)
        const auto &xHeader = reinterpret_cast<CTBNonModule &>(buffer);
        CHECK(xHeader.logicalName == "x.hpp" || xHeader.isHeaderUnit == false)

        BTCNonModule bigHu;
        bigHu.isHeaderUnit = true;
        bigHu.logicalNames.emplace_back("big.hpp");
        bigHu.logicalNames.emplace_back("y.hpp");
        bigHu.logicalNames.emplace_back("z.hpp");

        BMIFile bigHuBmi;
        bigHuBmi.filePath = bigPcm;

        CREATE_BMI_MAPPING(bigHuBmiMapping, bigHuBmi)
        bigHu.filePath = bigHuBmi.filePath;
        bigHu.fileSize = bigHuBmi.fileSize;

        SEND_MESSAGE(bigHu)

        readCompilerStdout(manager);
        CHECK(type == CTB::MODULE)
        const auto &aModule = reinterpret_cast<CTBModule &>(buffer);
        CHECK(aModule.moduleName == "A")

        BMIFile requested;
        requested.filePath = aPcm;
        CREATE_BMI_MAPPING(aPcmMapping, requested)

        BMIFile abModDepBmi;
        abModDepBmi.filePath = aBPcm;
        CREATE_BMI_MAPPING(aBPcmMapping, abModDepBmi)

        BMIFile acModDepBmi;
        acModDepBmi.filePath = aCPcm;
        CREATE_BMI_MAPPING(aCPcmMapping, acModDepBmi)

        BTCModule amod;
        amod.requested = requested;
        ModuleDep abModDep;
        abModDep.isHeaderUnit = false;
        abModDep.file = abModDepBmi;
        abModDep.logicalNames.emplace_back("A:B");
        amod.modDeps.emplace_back(std::move(abModDep));
        ModuleDep acModDep;
        acModDep.file = acModDepBmi;
        acModDep.logicalNames.emplace_back("A:C");
        amod.modDeps.emplace_back(std::move(acModDep));

        SEND_MESSAGE(amod)
        endCompilerTest();
        CLOSE_BMI_MAPPING(bigHuBmiMapping)
        CLOSE_BMI_MAPPING(aPcmMapping);
        CLOSE_BMI_MAPPING(aBPcmMapping);
        CLOSE_BMI_MAPPING(aCPcmMapping);
    }

    // compiling main.cpp
    {
        string compileCommand = CLANG_CMD R"( -std=c++20 -o ")" + mainObj + "\" -noScanIPC -c main.cpp";

        IPCManagerBS manager = readFirstCompilerStdout(compileCommand, true);

        CHECK(type == CTB::MODULE)
        const auto &ctbModule = reinterpret_cast<CTBModule &>(buffer);
        CHECK(ctbModule.moduleName == "Foo")

        BMIFile requested;
        requested.filePath = fooPcm;
        CREATE_BMI_MAPPING(requestedMapping, requested)

        BMIFile bigHuModDepBmi;
        bigHuModDepBmi.filePath = bigPcm;
        CREATE_BMI_MAPPING(bigHuModDepBmiMapping, bigHuModDepBmi)

        BMIFile aModDepBmi;
        aModDepBmi.filePath = aPcm;
        CREATE_BMI_MAPPING(aPcmMapping, aModDepBmi)

        BMIFile abModDepBmi;
        abModDepBmi.filePath = aBPcm;
        CREATE_BMI_MAPPING(aBPcmMapping, abModDepBmi)

        BMIFile acModDepBmi;
        acModDepBmi.filePath = aCPcm;
        CREATE_BMI_MAPPING(aCPcmMapping, acModDepBmi)

        BTCModule foo;
        foo.requested = requested;

        ModuleDep bigModDep;
        bigModDep.isHeaderUnit = true;
        bigModDep.file = bigHuModDepBmi;
        bigModDep.logicalNames.emplace_back("big.hpp");
        bigModDep.logicalNames.emplace_back("x.hpp");
        bigModDep.logicalNames.emplace_back("y.hpp");
        bigModDep.logicalNames.emplace_back("z.hpp");
        foo.modDeps.emplace_back(std::move(bigModDep));

        ModuleDep aModDep;
        aModDep.isHeaderUnit = false;
        aModDep.file = aModDepBmi;
        aModDep.logicalNames.emplace_back("A");
        foo.modDeps.emplace_back(std::move(aModDep));

        ModuleDep bModDep;
        bModDep.isHeaderUnit = false;
        bModDep.file = abModDepBmi;
        bModDep.logicalNames.emplace_back("A:B");
        foo.modDeps.emplace_back(std::move(bModDep));

        ModuleDep cModDep;
        cModDep.isHeaderUnit = false;
        cModDep.file = acModDepBmi;
        cModDep.logicalNames.emplace_back("A:C");
        foo.modDeps.emplace_back(std::move(cModDep));

        SEND_MESSAGE(foo)

        endCompilerTest();

        CLOSE_BMI_MAPPING(requestedMapping);
        CLOSE_BMI_MAPPING(bigHuModDepBmiMapping);
        CLOSE_BMI_MAPPING(aPcmMapping);
        CLOSE_BMI_MAPPING(aBPcmMapping);
        CLOSE_BMI_MAPPING(aCPcmMapping);
    }

    fflush(stdout);
    return {};
}
} // namespace

#ifdef IS_THIS_CLANG_REPO
TEST(IPC2978Test, IPC2978Test)
{
    const path p = current_path();
    current_path(LLVM_TOOLS_BINARY_DIR);
    const path mainFilePath = (LLVM_TOOLS_BINARY_DIR / path("main .o")).lexically_normal();
    remove(mainFilePath);

    const auto &r = runTest();
    current_path(p);
    if (!r)
    {
        FAIL() << r.error();
    }
    if (!exists(mainFilePath))
    {
        FAIL() << "main.o not found\n";
    }
}
#else
int main()
{
    remove(path("main .o"));
    if (const auto &r = runTest(); !r)
    {
        std::cout << r.error() << std::endl;
        return EXIT_FAILURE;
    }
    if (!exists(path("main .o")))
    {
        std::cout << "main.o not found" << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
#endif
