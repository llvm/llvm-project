// Shared process and protocol support for the library and Clang integration tests.
#ifndef IPC2978_TEST_PROCESS_HPP
#define IPC2978_TEST_PROCESS_HPP

#ifdef IS_THIS_CLANG_REPO
#include "clang/IPC2978/IPCManagerBS.hpp"
#else
#include "IPCManagerBS.hpp"
#endif
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#ifdef _WIN32
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <wordexp.h>
#endif

namespace ipc2978_test
{
inline bool endsWith(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Processes may run concurrently; a read waits for one selected child's next request or EOF.
class TestProcess
{
    static constexpr uint64_t invalid = UINT64_MAX;
    void (*onFailure)(const std::string &);

    bool fail(std::string message)
    {
        error = std::move(message);
        if (onFailure)
        {
            onFailure(error);
        }
        return false;
    }

    static void closePipe(uint64_t &pipe)
    {
        if (pipe == invalid)
        {
            return;
        }
#ifdef _WIN32
        CloseHandle(reinterpret_cast<HANDLE>(pipe));
#else
        close(static_cast<int>(pipe));
#endif
        pipe = invalid;
    }

  public:
    explicit TestProcess(void (*callback)(const std::string &) = nullptr) : onFailure(callback)
    {
    }
    TestProcess(const TestProcess &) = delete;
    TestProcess &operator=(const TestProcess &) = delete;

    ~TestProcess()
    {
        // An assertion may stop the test while the compiler is waiting for a response.
        if (pid != invalid)
        {
#ifdef _WIN32
            TerminateProcess(reinterpret_cast<HANDLE>(pid), 1);
            WaitForSingleObject(reinterpret_cast<HANDLE>(pid), INFINITE);
            CloseHandle(reinterpret_cast<HANDLE>(pid));
#else
            kill(static_cast<pid_t>(pid), SIGKILL);
            while (waitpid(static_cast<pid_t>(pid), nullptr, 0) == -1 && errno == EINTR)
            {
            }
#endif
        }
        closePipe(readPipe);
        closePipe(writePipe);
    }

    uint64_t pid = invalid;
    uint64_t readPipe = invalid;
    uint64_t writePipe = invalid;
    int exitStatus = -1;
    std::string error;

    bool startAsyncProcess(const char *command)
    {
        if (pid != invalid)
        {
            return fail("A test process is already running");
        }
        error.clear();
        exitStatus = -1;
#ifdef _WIN32
        SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE outputRead = nullptr, outputWrite = nullptr;
        HANDLE inputRead = nullptr, inputWrite = nullptr;
        if (!CreatePipe(&outputRead, &outputWrite, &attributes, 0))
        {
            return fail(P2978::getErrorString());
        }
        if (!CreatePipe(&inputRead, &inputWrite, &attributes, 0))
        {
            const auto message = P2978::getErrorString();
            CloseHandle(outputRead);
            CloseHandle(outputWrite);
            return fail(message);
        }
        readPipe = reinterpret_cast<uint64_t>(outputRead);
        writePipe = reinterpret_cast<uint64_t>(inputWrite);
        if (!SetHandleInformation(outputRead, HANDLE_FLAG_INHERIT, 0) ||
            !SetHandleInformation(inputWrite, HANDLE_FLAG_INHERIT, 0))
        {
            const auto message = P2978::getErrorString();
            CloseHandle(inputRead);
            CloseHandle(outputWrite);
            closePipe(readPipe);
            closePipe(writePipe);
            return fail(message);
        }
        STARTUPINFOA startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = inputRead;
        startup.hStdOutput = outputWrite;
        startup.hStdError = outputWrite;
        PROCESS_INFORMATION process{};
        std::string mutableCommand(command);
        const bool started = CreateProcessA(nullptr, mutableCommand.data(), nullptr, nullptr, TRUE,
                                            CREATE_NEW_PROCESS_GROUP, nullptr, nullptr, &startup, &process);
        const auto message = started ? std::string() : P2978::getErrorString();
        CloseHandle(inputRead);
        CloseHandle(outputWrite);
        if (!started)
        {
            closePipe(readPipe);
            closePipe(writePipe);
            return fail(message);
        }
        CloseHandle(process.hThread);
        pid = reinterpret_cast<uint64_t>(process.hProcess);
#else
        int output[2], input[2];
        if (pipe(output) == -1)
        {
            return fail(P2978::getErrorString());
        }
        if (pipe(input) == -1)
        {
            const auto message = P2978::getErrorString();
            close(output[0]);
            close(output[1]);
            return fail(message);
        }
        // A later child must not inherit the pipe ends of an earlier child.
        for (int fd : {output[0], output[1], input[0], input[1]})
        {
            if (fcntl(fd, F_SETFD, FD_CLOEXEC) == -1)
            {
                const auto message = P2978::getErrorString();
                for (int pipeFd : {output[0], output[1], input[0], input[1]})
                {
                    close(pipeFd);
                }
                return fail(message);
            }
        }
        const pid_t child = fork();
        if (child == 0)
        {
            if (dup2(input[0], STDIN_FILENO) == -1 || dup2(output[1], STDOUT_FILENO) == -1 ||
                dup2(output[1], STDERR_FILENO) == -1)
            {
                _exit(127);
            }
            for (int fd : {output[0], output[1], input[0], input[1]})
            {
                close(fd);
            }
            wordexp_t words{};
            if (wordexp(command, &words, WRDE_NOCMD) != 0 || words.we_wordc == 0)
            {
                _exit(127);
            }
            execvp(words.we_wordv[0], words.we_wordv);
            perror("execvp");
            _exit(127);
        }
        const auto message = child == -1 ? P2978::getErrorString() : std::string();
        close(output[1]);
        close(input[0]);
        readPipe = output[0];
        writePipe = input[1];
        if (child == -1)
        {
            closePipe(readPipe);
            closePipe(writePipe);
            return fail(message);
        }
        pid = child;
#endif
        return true;
    }

    bool reapProcess()
    {
        if (pid == invalid)
        {
            return fail("No test process to reap");
        }
#ifdef _WIN32
        if (WaitForSingleObject(reinterpret_cast<HANDLE>(pid), INFINITE) != WAIT_OBJECT_0)
        {
            return fail(P2978::getErrorString());
        }
        DWORD status = 0;
        if (!GetExitCodeProcess(reinterpret_cast<HANDLE>(pid), &status))
        {
            return fail(P2978::getErrorString());
        }
        exitStatus = static_cast<int>(status);
        CloseHandle(reinterpret_cast<HANDLE>(pid));
#else
        int status;
        pid_t result;
        do
        {
            result = waitpid(static_cast<pid_t>(pid), &status, 0);
        } while (result == -1 && errno == EINTR);
        if (result == -1)
        {
            return fail(P2978::getErrorString());
        }
        exitStatus = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
        pid = invalid;
        closePipe(readPipe);
        closePipe(writePipe);
        return true;
    }

    // True means a complete protocol frame. False means EOF or an error recorded in error.
    bool readCompilerMessage(std::string &output)
    {
        if (readPipe == invalid)
        {
            return fail("No test process output pipe");
        }
        while (true)
        {
            char buffer[4096];
#ifdef _WIN32
            DWORD count = 0;
            if (!ReadFile(reinterpret_cast<HANDLE>(readPipe), buffer, sizeof(buffer), &count, nullptr))
            {
                if (GetLastError() == ERROR_BROKEN_PIPE)
                {
                    return false;
                }
                return fail(P2978::getErrorString());
            }
#else
            const uint64_t count = read(static_cast<int>(readPipe), buffer, sizeof(buffer));
            if (count == UINT64_MAX)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return fail(P2978::getErrorString());
            }
#endif
            if (count == 0)
            {
                return false;
            }
            output.append(buffer, count);
            if (endsWith(output, P2978::delimiter))
            {
                return true;
            }
        }
    }

    bool pruneCompilerOutput(std::string &output, char (&buffer)[320], P2978::CTB &type)
    {
        const uint64_t trailerSize = sizeof(uint32_t) + strlen(P2978::delimiter);
        if (output.size() < trailerSize)
        {
            return fail("Received IPC frame without a payload size");
        }
        uint32_t payloadSize;
        memcpy(&payloadSize, output.data() + output.size() - trailerSize, sizeof(payloadSize));
        if (payloadSize > output.size() - trailerSize)
        {
            return fail("Received IPC payload size exceeds the available bytes");
        }
        const char *payload = output.data() + output.size() - trailerSize - payloadSize;
        if (const auto result = P2978::IPCManagerBS::receiveMessage(buffer, type, {payload, payloadSize}); !result)
        {
            return fail(result.error());
        }
        output.resize(output.size() - trailerSize - payloadSize);
        return true;
    }
};
} // namespace ipc2978_test
#endif
