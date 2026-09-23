
#ifndef IPC_MANAGER_BS_HPP
#define IPC_MANAGER_BS_HPP

#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"

namespace P2978
{

// Parses dependency requests received by the build system.
class IPCManagerBS
{
  public:
    // Parse one payload after the caller removes diagnostics, payload size, and delimiter.
    // ctbBuffer must be aligned for CTBModule/CTBNonModule. Parsed string views borrow
    // serverReadString, whose bytes must remain alive until the request is consumed.
    static Result<void> receiveMessage(char (&ctbBuffer)[320], CTB &messageType, std::string_view serverReadString);
};
} // namespace P2978
#endif // IPC_MANAGER_BS_HPP
