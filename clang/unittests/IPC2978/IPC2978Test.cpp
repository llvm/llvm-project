//===- unittests/IPC2978/IPC2978Test.cpp - Tests IPC2978 Support -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The following line is uncommented by clang/lib/IPC2978/setup.py for clang/unittests/IPC2978/IPC2978.cpp

#define IS_THIS_CLANG_REPO
#ifdef IS_THIS_CLANG_REPO
#include "clang/IPC2978/IPCManagerBS.hpp"
#include "gtest/gtest.h"
template <typename T> void printMessage(const T &, bool)
{
}
#else
#include "IPCManagerBS.hpp"
#include "Testing.hpp"
#include "fmt/printf.h"
#endif

#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace std::filesystem;
using namespace N2978;
using namespace std;

#ifdef _WIN32
#define CLANG_CMD ".\\clang.exe"
#else
#define CLANG_CMD "./clang"
#endif

namespace
{
#ifdef _WIN32
PROCESS_INFORMATION pi;
tl::expected<void, string> Run(const string &command)
{
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    if (!CreateProcessA(nullptr,                             // lpApplicationName
                        const_cast<char *>(command.c_str()), // lpCommandLine
                        nullptr,                             // lpProcessAttributes
                        nullptr,                             // lpThreadAttributes
                        FALSE,                               // bInheritHandles
                        0,                                   // dwCreationFlags
                        nullptr,                             // lpEnvironment
                        nullptr,                             // lpCurrentDirectory
                        &si,                                 // lpStartupInfo
                        &pi                                  // lpProcessInformation
                        ))
    {
        return tl::unexpected("CreateProcess" + getErrorString());
    }
    return {};
}
#else

int procStatus;
int procId;
/// Start a process and gather its raw output.  Returns its exit code.
/// Crashes (calls Fatal()) on error.
tl::expected<void, string> Run(const string &command)
{
    if (procId = fork(); procId == -1)
    {
        return tl::unexpected("fork" + getErrorString());
    }
    if (procId == 0)
    {
        // Child process
        exit(WEXITSTATUS(system(command.c_str())));
    }
    return {};
}
#endif

tl::expected<void, std::string> CloseProcess()
{
#ifdef _WIN32
    if (WaitForSingleObject(pi.hProcess, INFINITE) == WAIT_FAILED)
    {
        return tl::unexpected("WaitForSingleObject" + getErrorString());
    }
#else
    if (waitpid(procId, &procStatus, 0) == -1)
    {
        return tl::unexpected("waitpid" + getErrorString());
    }
#endif
    return {};
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
// has no closing brace to test for error.

)";

// Creates all the input files (source files + pcm files) that are needed for the test.
void setupTest()
{
    //  A.cpp
    const string aDotCpp = R"(
export module A;     // primary module interface unit

export import :B;    // Hello() is visible when importing 'A'.
import :C;           // WorldImpl() is now visible only for 'A.cpp'.
// export import :C; // ERROR: Cannot export a module implementation unit.

// World() is visible by any translation unit importing 'A'.
export char const* World()
{
    return WorldImpl();
}
)";
    // A-B.cpp
    const string aBDotCPP = R"(
export module A:B; // partition module interface unit

// Hello() is visible by any translation unit importing 'A'.
export char const* Hello() { return "Hello"; }
)";

    // A-C.cpp
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

    // Foo.cpp
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

    ofstream("A.cpp") << aDotCpp;
    ofstream("A-B.cpp") << aBDotCPP;
    ofstream("A-C.cpp") << aCDotCPP;
    ofstream("m.hpp") << mDotHpp;
    ofstream("n.hpp") << nDotHpp;
    ofstream("o.hpp") << oDotHpp;
    ofstream("x.hpp") << xDotHpp;
    ofstream("y.hpp") << yDotHpp;
    ofstream("z.hpp") << zDotHpp;
    ofstream("big.hpp") << bigDotHpp;
    ofstream("Foo.cpp") << fooDotCpp;
    ofstream("main.cpp") << mainDotCpp;
}

tl::expected<int, string> runTest()
{
    setupTest();

    string current = current_path().generic_string() + '/';
    string mainFilePath = current + "main .o";
    string modFilePath = current + "mod .pcm";
    string mod1FilePath = current + "mod1 .pcm";
    string mod2FilePath = current + "mod2 .pcm";

    string aCObj = (current_path() / "A-C .o").generic_string();
    string aCPcm = (current_path() / "A-C .pcm").generic_string();
    string aBObj = (current_path() / "A-B .o").generic_string();
    string aBPcm = (current_path() / "A-B .pcm").generic_string();
    string aObj = (current_path() / "A .o").generic_string();
    string aPcm = (current_path() / "A .pcm").generic_string();
    string bObj = (current_path() / "B .o").generic_string();
    string bPcm = (current_path() / "B .pcm").generic_string();
    string mHpp = (current_path() / "m.hpp").generic_string();
    string nHpp = (current_path() / "n.hpp").generic_string();
    string oHpp = (current_path() / "o.hpp").generic_string();
    string nPcm = (current_path() / "N .pcm").generic_string();
    string oPcm = (current_path() / "O .pcm").generic_string();
    string xHpp = (current_path() / "x.hpp").generic_string();
    string yHpp = (current_path() / "y.hpp").generic_string();
    string zHpp = (current_path() / "z.hpp").generic_string();
    string bigHpp = (current_path() / "big.hpp").generic_string();
    string bigPcm = (current_path() / "Big .pcm").generic_string();
    string fooPcm = (current_path() / "Foo .pcm").generic_string();
    string fooObj = (current_path() / "Foo .o").generic_string();
    string mainObj = (current_path() / "main .o").generic_string();

    // compiling A-C.cpp
    {
        const auto &r = makeIPCManagerBS(aCObj);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + aCObj +
                                "\" -noScanIPC -c -xc++-module A-C.cpp -fmodule-output=\"" + aCPcm + "\"";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);

        if (ctbLastMessage.logicalName != "A:C")
        {
            return tl::unexpected("wrong logical name received while compiling A-C.cpp");
        }
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling A-B.cpp
    {
        const auto &r = makeIPCManagerBS(aBObj);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + aBObj +
                                "\" -noScanIPC -c -xc++-module A-B.cpp -fmodule-output=\"" + aBPcm + "\"";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);

        if (ctbLastMessage.logicalName != "A:B")
        {
            return tl::unexpected("wrong logical name received while compiling A-B.cpp");
        }
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling A.cpp
    {
        const auto &r = makeIPCManagerBS(aObj);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + aObj +
                                "\" -noScanIPC -c -xc++-module A.cpp -fmodule-output=\"" + aPcm + "\"";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbModule = reinterpret_cast<CTBModule &>(buffer);

        if (ctbModule.moduleName != "A:B")
        {
            return tl::unexpected("wrong logical name received while compiling A-B.cpp");
        }
        printMessage(ctbModule, false);

        BTCModule btcMod;
        btcMod.requested.filePath = aBPcm;
        ModuleDep modDep;
        modDep.file.filePath = aCPcm;
        modDep.logicalNames.emplace_back("A:C");
        modDep.isHeaderUnit = false;
        btcMod.modDeps.emplace_back(std::move(modDep));

        if (const auto &r2 = manager.sendMessage(std::move(btcMod)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling n.hpp
    {
        const auto &r = makeIPCManagerBS(nPcm);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + nPcm +
                                "\" -noScanIPC -xc++-header n.hpp -DCOMMAND_MACRO";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);

        if (ctbNonModMHpp.logicalName != "m.hpp" || ctbNonModMHpp.isHeaderUnit == true)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule nonModMPcm;
        nonModMPcm.isHeaderUnit = false;
        nonModMPcm.filePath = mHpp;
        if (const auto &r2 = manager.sendMessage(std::move(nonModMPcm)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling o.hpp
    {
        const auto &r = makeIPCManagerBS(oPcm);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand =
            CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + oPcm + "\" -noScanIPC -xc++-header o.hpp";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);
        if (ctbNonModMHpp.logicalName != "m.hpp" || ctbNonModMHpp.isHeaderUnit == true)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule nonModMPcm;
        nonModMPcm.isHeaderUnit = false;
        nonModMPcm.filePath = mHpp;
        if (const auto &r2 = manager.sendMessage(std::move(nonModMPcm)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbNonModNHpp = reinterpret_cast<CTBNonModule &>(buffer);
        if (ctbNonModNHpp.logicalName != "n.hpp" || ctbNonModNHpp.isHeaderUnit == false)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule nonModNPcm;
        nonModNPcm.isHeaderUnit = true;
        nonModNPcm.filePath = nPcm;

        if (const auto &r2 = manager.sendMessage(std::move(nonModNPcm)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling o.hpp with include-translation. BTCNonModule for n.hpp will be received with
    // isHeaderUnit = true.
    {
        const auto &r = makeIPCManagerBS(oPcm);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + oPcm +
                                "\" -noScanIPC -xc++-header o.hpp -DTRANSLATING";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);
        if (ctbNonModMHpp.logicalName != "m.hpp" || ctbNonModMHpp.isHeaderUnit == true)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule nonModMPcm;
        nonModMPcm.isHeaderUnit = false;
        nonModMPcm.filePath = mHpp;
        if (const auto &r2 = manager.sendMessage(std::move(nonModMPcm)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbNonModNHpp = reinterpret_cast<CTBNonModule &>(buffer);
        if (ctbNonModNHpp.logicalName != "n.hpp" || ctbNonModNHpp.isHeaderUnit == true)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule nonModNPcm;
        nonModNPcm.isHeaderUnit = true;
        nonModNPcm.filePath = nPcm;

        if (const auto &r2 = manager.sendMessage(std::move(nonModNPcm)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling big.hpp
    {
        const auto &r = makeIPCManagerBS(bigPcm);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand =
            CLANG_CMD R"( -std=c++20 -fmodule-header=user -o ")" + bigPcm + "\" -noScanIPC -xc++-header big.hpp";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &ctbNonModMHpp = reinterpret_cast<CTBNonModule &>(buffer);

        if (ctbNonModMHpp.logicalName != "x.hpp" || ctbNonModMHpp.isHeaderUnit == true)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule headerFile;
        headerFile.isHeaderUnit = false;
        headerFile.filePath = xHpp;
        HeaderFile yHeaderFile;
        yHeaderFile.logicalName = "y.hpp";
        yHeaderFile.filePath = yHpp;
        yHeaderFile.isSystem = true;
        headerFile.headerFiles.emplace_back(std::move(yHeaderFile));
        HeaderFile zHeaderFile;
        zHeaderFile.logicalName = "z.hpp";
        zHeaderFile.filePath = zHpp;
        zHeaderFile.isSystem = true;
        headerFile.headerFiles.emplace_back(std::move(zHeaderFile));

        if (const auto &r2 = manager.sendMessage(std::move(headerFile)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling Foo.cpp
    {
        const auto &r = makeIPCManagerBS(fooObj);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -fmodules-reduced-bmi -o ")" + fooObj +
                                "\" -noScanIPC -c -xc++-module Foo.cpp -fmodule-output=\"" + fooPcm + "\"";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::NON_MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &xHeader = reinterpret_cast<CTBNonModule &>(buffer);

        if (xHeader.logicalName != "x.hpp" || xHeader.isHeaderUnit == true)
        {
            return tl::unexpected("wrong message received");
        }

        BTCNonModule bigHu;
        bigHu.isHeaderUnit = true;
        bigHu.filePath = bigPcm;
        bigHu.logicalNames.emplace_back("big.hpp");
        bigHu.logicalNames.emplace_back("y.hpp");
        bigHu.logicalNames.emplace_back("z.hpp");

        if (const auto &r2 = manager.sendMessage(bigHu); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }
        const auto &aModule = reinterpret_cast<CTBModule &>(buffer);

        if (aModule.moduleName != "A")
        {
            return tl::unexpected("wrong message received");
        }

        BTCModule amod;
        amod.requested.filePath = aPcm;
        ModuleDep abModDep;
        abModDep.isHeaderUnit = false;
        abModDep.file.filePath = aBPcm;
        abModDep.logicalNames.emplace_back("A:B");
        amod.modDeps.emplace_back(std::move(abModDep));
        ModuleDep acModDep;
        acModDep.file.filePath = aCPcm;
        acModDep.logicalNames.emplace_back("A:C");
        amod.modDeps.emplace_back(std::move(acModDep));

        if (const auto &r2 = manager.sendMessage(amod); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
    }

    // compiling main.cpp
    auto compileMain = [&](bool shouldFail) -> tl::expected<int, string> {
        const auto &r = makeIPCManagerBS(mainObj);
        if (!r)
        {
            return tl::unexpected("creating manager failed" + r.error() + "\n");
        }

        const IPCManagerBS &manager = *r;

        string compileCommand = CLANG_CMD R"( -std=c++20 -o ")" + mainObj + "\" -noScanIPC -c main.cpp";
        if (const auto &r2 = Run(compileCommand); !r2)
        {
            return tl::unexpected(r2.error());
        }

        CTB type;
        char buffer[320];
        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::MODULE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbModule = reinterpret_cast<CTBModule &>(buffer);

        if (ctbModule.moduleName != "Foo")
        {
            return tl::unexpected("wrong logical name received while compiling A-B.cpp");
        }
        printMessage(ctbModule, false);

        BTCModule m;
        m.requested.filePath = fooPcm;
        ModuleDep modDep;
        modDep.file.filePath = bigPcm;
        modDep.logicalNames.emplace_back("big.hpp");
        modDep.logicalNames.emplace_back("x.hpp");
        modDep.logicalNames.emplace_back("y.hpp");
        modDep.logicalNames.emplace_back("z.hpp");
        modDep.isHeaderUnit = true;
        m.modDeps.emplace_back(std::move(modDep));

        ModuleDep aModDep;
        aModDep.isHeaderUnit = false;
        aModDep.file.filePath = aPcm;
        aModDep.logicalNames.emplace_back("A");
        m.modDeps.emplace_back(std::move(aModDep));

        ModuleDep bModDep;
        bModDep.isHeaderUnit = false;
        bModDep.file.filePath = aBPcm;
        bModDep.logicalNames.emplace_back("A:B");
        m.modDeps.emplace_back(std::move(bModDep));

        ModuleDep cModDep;
        cModDep.isHeaderUnit = false;
        cModDep.file.filePath = aCPcm;
        cModDep.logicalNames.emplace_back("A:C");
        m.modDeps.emplace_back(std::move(cModDep));

        if (const auto &r2 = manager.sendMessage(std::move(m)); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager send message failed" + r2.error() + "\n");
        }

        if (const auto &r2 = manager.receiveMessage(buffer, type); !r2)
        {
            string str = r2.error();
            return tl::unexpected("manager receive message failed" + r2.error() + "\n");
        }

        if (type != CTB::LAST_MESSAGE)
        {
            return tl::unexpected("received message of wrong type");
        }

        const auto &ctbLastMessage = reinterpret_cast<CTBLastMessage &>(buffer);
        if (ctbLastMessage.errorOccurred != shouldFail)
        {
            return tl::unexpected("wrong last message received");
        }

        printMessage(ctbLastMessage, false);
        manager.closeConnection();
        if (const auto &r2 = CloseProcess(); !r2)
        {
            return tl::unexpected("closing process failed");
        }
        return {};
    };

    if (const auto &r = compileMain(true); !r)
    {
        string str = r.error();
        return tl::unexpected("compiling main failed" + r.error() + "\n");
    }
    // main.cpp
    ofstream("main.cpp") << mainDotCpp + '}';

    if (const auto &r = compileMain(false); !r)
    {
        string str = r.error();
        return tl::unexpected("compiling main failed" + r.error() + "\n");
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
        fmt::print("{}\n", r.error());
        return EXIT_FAILURE;
    }
    if (!exists(path("main .o")))
    {
        fmt::print("main.o not found\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
#endif
