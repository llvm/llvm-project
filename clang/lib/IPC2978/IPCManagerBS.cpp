#include "clang/IPC2978/IPCManagerBS.hpp"
#define TRY_READ_VAL(var, func, ...)                                                                                   \
    const auto &var##_result = func(__VA_ARGS__);                                                                      \
    if (!var##_result)                                                                                                 \
    {                                                                                                                  \
        return Error{var##_result.error()};                                                                            \
    }                                                                                                                  \
    auto &var = *var##_result;

namespace P2978
{

Result<void> IPCManagerBS::receiveMessage(char (&ctbBuffer)[320], CTB &messageType,
                                          const std::string_view serverReadString)
{
    if (serverReadString.empty())
    {
        return Error{getErrorString(ErrorCategory::PARSING_ERROR)};
    }

    uint64_t bytesRead = 1;

    // The nonempty payload starts with a one-byte request type.
    switch (static_cast<CTB>(serverReadString[0]))
    {

    case CTB::MODULE: {
        TRY_READ_VAL(r, Manager::readString, serverReadString, bytesRead);

        messageType = CTB::MODULE;
        getInitializedObjectFromBuffer<CTBModule>(ctbBuffer).moduleName = r;
    }
    break;

    case CTB::NON_MODULE: {
        TRY_READ_VAL(r, Manager::readBool, serverReadString, bytesRead);
        TRY_READ_VAL(r2, Manager::readString, serverReadString, bytesRead);
        messageType = CTB::NON_MODULE;
        auto &[isHeaderUnit, str] = getInitializedObjectFromBuffer<CTBNonModule>(ctbBuffer);
        isHeaderUnit = r;
        str = r2;
    }
    break;

    default:
        return Error{getErrorString(ErrorCategory::UNKNOWN_CTB_TYPE)};
    }

    if (serverReadString.size() != bytesRead)
    {
        return Error{getErrorString(serverReadString.size(), bytesRead)};
    }

    return {};
}

} // namespace P2978
